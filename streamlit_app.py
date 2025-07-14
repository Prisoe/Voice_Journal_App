import os
import boto3
import openai
import faiss
import numpy as np
import streamlit as st
import pickle
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# AWS DynamoDB config
dynamodb = boto3.resource(
    'dynamodb',
    region_name=os.getenv('AWS_REGION'),
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
)
table = dynamodb.Table(os.getenv('DYNAMODB_TABLE_NAME'))

# File paths
INDEX_FILE = "faiss_index.bin"
DOCS_FILE = "embedded_docs.pkl"
EMBEDS_FILE = "embedded_vectors.npy"

# Load transcripts from DynamoDB
def load_transcripts():
    response = table.scan()
    items = response['Items']
    docs = [item['transcription'] for item in items if 'transcription' in item]
    return docs

# Get OpenAI embedding
def get_embedding(text):
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# Build and save vector index
def build_and_save_index(docs):
    embeddings = [get_embedding(doc) for doc in docs]
    vectors = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(1536)
    index.add(vectors)

    with open(DOCS_FILE, "wb") as f:
        pickle.dump(docs, f)
    with open(EMBEDS_FILE, "wb") as f:
        np.save(f, vectors)
    faiss.write_index(index, INDEX_FILE)

    return index, docs, vectors

# Load vector index from disk
def load_index():
    if not os.path.exists(INDEX_FILE):
        return None, None, None
    index = faiss.read_index(INDEX_FILE)
    with open(DOCS_FILE, "rb") as f:
        docs = pickle.load(f)
    vectors = np.load(EMBEDS_FILE)
    return index, docs, vectors

# Query the RAG system
def query_rag(question, docs, vectors, index):
    q_embed = np.array(get_embedding(question)).astype("float32")
    D, I = index.search(np.array([q_embed]), k=3)
    context = "\n\n".join([docs[i] for i in I[0]])

    prompt = f"""You are a journal assistant. Based on the entries below, answer the question.

Journal Entries:
{context}

User Question: {question}
Answer:"""

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# Streamlit UI
st.set_page_config(page_title="Voice Journal RAG")
st.title(" Voice Journal Query Assistant")

# Load or build index
if "index" not in st.session_state:
    index, docs, vectors = load_index()
    if index:
        st.session_state.index = index
        st.session_state.docs = docs
        st.session_state.vectors = vectors
    else:
        st.warning("No FAISS index found. Click 'Refresh Index' to create one.")

# Query interface
if "index" in st.session_state:
    question = st.text_input("Ask something about your past journal entries:")
    if st.button("Ask") and question:
        answer = query_rag(question, st.session_state.docs, st.session_state.vectors, st.session_state.index)
        st.markdown(f"**Answer:** {answer}")

# Refresh button
if st.button(" Refresh Index (rebuild from DynamoDB)"):
    with st.spinner("Rebuilding index..."):
        docs = load_transcripts()
        index, docs, vectors = build_and_save_index(docs)
        st.session_state.index = index
        st.session_state.docs = docs
        st.session_state.vectors = vectors
    st.success("Index refreshed!")
