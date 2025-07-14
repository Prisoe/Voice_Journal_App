# 🎙️ Voice Journaling App (RAG-Enabled)

This project is a voice-first journaling system that allows users to **record audio journals**, automatically transcribe them, store them in the cloud, and later **query those journal entries using natural language** — powered by Retrieval-Augmented Generation (RAG) and GPT-4.

---

## 🚀 Features

- 🎙 **Voice Journal Upload:** Record or upload voice notes (instead of typing)
- 🧾 **Automatic Transcription:** Audio is transcribed using **Amazon Transcribe**
- ☁ **Cloud Storage:**
  - Audio stored in **Amazon S3**
  - Transcripts saved in **DynamoDB**
- 🧠 **Query Your Life:** Ask natural questions like:
  > _"What did I say I forget to buy two weeks ago?"_  
  > _"Why was I stressed last Thursday?"_
- 🔍 **RAG Integration:**
  - Embeds transcripts using **OpenAI embeddings**
  - Stores them in a **FAISS** vector index
  - Retrieves the most relevant entries to build GPT-4 prompts

---

## 📂 Tech Stack

| Component        | Technology                    |
|------------------|-------------------------------|
| Backend          | Flask + Streamlit             |
| Transcription    | AWS Transcribe                |
| Storage          | AWS S3 + DynamoDB             |
| Embeddings       | OpenAI `text-embedding-3-small` |
| RAG Model        | GPT-4 via OpenAI Chat API     |
| Vector Search    | FAISS                         |

---

## 🛠 Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/Prisoe/Voice_Journal_App.git
cd "Voice_Journal_App/Simple Voice Journalling App"


cp .env.example .env

pip install -r requirements.txt

python app.py

streamlit run streamlit_app.py
