import os
import boto3
import openai
import faiss
import numpy as np
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# DynamoDB setup
dynamodb = boto3.resource(
    'dynamodb',
    region_name=os.getenv('AWS_REGION'),
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
)
table = dynamodb.Table(os.getenv('DYNAMODB_TABLE_NAME'))



def load_transcripts():
    response = table.scan()
    items = response['Items']
    docs = [item['transcription'] for item in items if 'transcription' in item]
    return docs



def get_embedding(text):
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def build_vector_store(docs):
    index = faiss.IndexFlatL2(1536)  # 1536 dims for text-embedding-3-small
    embeddings = [get_embedding(doc) for doc in docs]
    index.add(np.array(embeddings).astype("float32"))
    return index, docs, embeddings



def query_rag(user_question, docs, embeddings, index):
    q_embed = np.array(get_embedding(user_question)).astype("float32")
    D, I = index.search(np.array([q_embed]), k=3)  # Top 3 matches
    context = "\n\n".join([docs[i] for i in I[0]])

    prompt = f"""You are a journal assistant. Based on the entries below, answer the question.

Journal Entries:
{context}

User Question: {user_question}
Answer:"""

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()



if __name__ == "__main__":
    print("Loading transcripts from DynamoDB...")
    documents = load_transcripts()
    print(f"Loaded {len(documents)} entries.")

    print("Building vector index...")
    index, docs, embeds = build_vector_store(documents)

    while True:
        q = input("\nAsk something from your journal history: ")
        if q.lower() in ['exit', 'quit']:
            break
        print("\n Searching...")
        answer = query_rag(q, docs, embeds, index)
        print(f"\n Answer: {answer}")
