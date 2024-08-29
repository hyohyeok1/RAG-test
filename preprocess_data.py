from transformers import AutoTokenizer, AutoModel
import torch
import docx
import re

# Function to extract text from .docx file
def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

# Function to split text into sections based on "제X조" pattern
def split_into_sections(text):
    sections = []
    current_text = []
    special_char_pattern = re.compile(r'[^0-9a-zA-Z가-힣.\s]')
    
    for line in text.splitlines():
        if line.startswith("제") and "조" in line:
            # Store the previous section if it exists
            if current_text:
                sections.append(' '.join(current_text).strip())
            # Start a new section
            current_text = [special_char_pattern.sub(' ', line.strip())]
        else:
            clean_line = special_char_pattern.sub(' ', line)
            if clean_line:
                current_text.append(clean_line.strip())
    
    # Store the last section if it exists
    if current_text:
        sections.append(' '.join(current_text).strip())
    
    return sections

# Function to get embedding vector for a given text
def get_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1)    
    # Apply L2 normalization
    norm = torch.norm(embedding, p=2, dim=1, keepdim=True)
    normalized_embedding = embedding / norm
    return normalized_embedding

def preprocess_data():
	# Load pre-trained BERT model and tokenizer
	tokenizer = AutoTokenizer.from_pretrained('jhgan/ko-sroberta-multitask')
	model = AutoModel.from_pretrained('jhgan/ko-sroberta-multitask')

	# Path to the uploaded .docx file
	file_path = './constitution.docx'

	# Extract and process text
	text = extract_text_from_docx(file_path)
	sections = split_into_sections(text)

	# Initialize a list to store the embeddings
	data = []

	# Get embeddings for each section
	for section_text in sections:
		embedding = get_embedding(section_text, tokenizer, model)
		embedding_list = embedding.squeeze().numpy().tolist()
		data.append({"text": section_text, "embedding": embedding_list})
	
	return data
