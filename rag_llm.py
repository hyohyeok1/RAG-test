import gradio as gr
from preprocess_data import get_embedding, preprocess_data
from pymilvus import FieldSchema, CollectionSchema, DataType, MilvusClient
from transformers import AutoTokenizer, AutoModel
from dotenv import load_dotenv

from openai import OpenAI

openai_client = OpenAI()

load_dotenv()

def create_schema(): 
    fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),  # Primary Key
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),  # BERT 임베딩 크기
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=5000)  # 텍스트 필드
    ]
    
    schema = CollectionSchema(fields=fields, description="Constitution Sections")
    return schema

def create_index_params():
    index_params = MilvusClient.prepare_index_params()
    
    index_params.add_index(field_name="embedding", metric_type="IP", index_type="IVF_FLAT", index_name="vector_index", params={"nlist": 128})
    return index_params

# Milvus에 연결
client = MilvusClient(uri="http://localhost:19530")

# Milvus에서 Collection 설정
collection_name = "constitution_sections"

if client.has_collection(collection_name=collection_name):
    client.drop_collection(collection_name=collection_name)

schema = create_schema()
index_params = create_index_params()

client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)

data = preprocess_data()
client.insert(
    collection_name=collection_name,
    data=data)

tokenizer = AutoTokenizer.from_pretrained('jhgan/ko-sroberta-multitask')
model = AutoModel.from_pretrained('jhgan/ko-sroberta-multitask')

def search_constitution(query_text):
    
	embedding_query = (get_embedding(query_text, tokenizer, model)).numpy().tolist()

	search_res = client.search(collection_name=collection_name,
						data=embedding_query,
						anns_field="embedding",
						limit=10,
						search_params={"metric_type": "IP", "params": {"level": 3}},
						output_fields=["text"],
                        )

	retrieved_lines = [
		res["entity"]["text"] for res in search_res[0]
	]

	context = "\n".join(
		[line_with_distance for line_with_distance in retrieved_lines]
	)
     
	print(context)

	SYSTEM_PROMPT = """
	사람: You are an AI assistant. You are able to find answers to the questions from the contextual passage snippets provided.
	"""
	USER_PROMPT = f"""
	사용자가 입력한 헌법 조항에 따라, 아래에 제공된 <context> 태그로 감싸인 정보 조각을 사용하여 질문에 대한 답변을 제공해 주세요.
	<context>
	{context}
	</context>
	<question>
	사용자가 입력한 헌법 조항에 해당하는 내용을 찾아주세요: {query_text}
	</question>
	"""

	response = openai_client.chat.completions.create(
		model="gpt-4o-mini",
		messages=[
			{"role": "system", "content": SYSTEM_PROMPT},
			{"role": "user", "content": USER_PROMPT},
		],
	)
	return response.choices[0].message.content

# Define the Gradio interface
interface = gr.Interface(
    fn=search_constitution,
    inputs=gr.Textbox(
        label="질문을 입력하세요",
        placeholder="예: 제 23조 법률의 항목이 뭐야?",
        lines=5
    ),  
    examples=[
        "헌법 제 23조 법률의 항목이 뭐야?",
        "헌법 제 46조는 무엇인가요?",
        "헌법 제 69조는 어떤 내용을 담고 있나요?"
    ],
    outputs="text",
    title="헌법 조항 검색",
    description="헌법 조항을 검색하고 질문에 답변합니다. 헌법은 제 1조부터 130조까지 있습니다."
)

# Launch the interface
if __name__ == "__main__":
    interface.launch(share=True)