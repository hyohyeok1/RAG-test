# RAG-test

## Requirement

### Milvus 설치

```sh
bash standalone_embed.sh start
```
위 명령어를 통해 Milvus DB Docker Image를 설치하고 실행합니다.

```sh
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh

bash standalone_embed.sh start
```
실행되지 않는다면 위의 명령어를 통해 설치하고 실행시킬 수 있습니다.

### OPENAI API Key 준비
위의 소스코드는 `ChatGPT-4o-mini`를 사용합니다. `.env` 파일을 만들어 `OPENAI_API_KEY="your_api_key"` 형태로 API Key를 추가해줍니다.

### 패키지 설치

```sh
pip install -r requirements.txt
```
위 명령어를 실행해 패키지를 설치합니다.

## 실행

```sh
python rag_llm.py
```
위 명령어를 통해 실행시킵니다.