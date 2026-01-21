import os
import uuid
import time
import pickle
from datetime import datetime, date
from pathlib import Path

import boto3
import numpy as np
import requests
import faiss
import streamlit as st
import openai
from dotenv import load_dotenv
from botocore.exceptions import ClientError

# =========================================================
# Streamlit native recording (NO pydub / NO audiorecorder)
# ✅ Works on Render without PyAudio/PortAudio/FFmpeg headaches
# =========================================================

# -------------------------
# Config
# -------------------------
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = os.getenv("TRANSCRIBE_BUCKET_NAME")
DDB_TABLE = os.getenv("DYNAMODB_TABLE_NAME")

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
EMBED_DIM = 1536

APP_PASSWORD = os.getenv("APP_PASSWORD", "")

INDEX_FILE = "faiss_index.bin"
DOCS_FILE = "embedded_docs.pkl"
META_FILE = "embedded_meta.pkl"
VECTORS_FILE = "embedded_vectors.npy"

# -------------------------
# AWS clients
# -------------------------
s3 = boto3.client(
    "s3",
    region_name=AWS_REGION,
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)

transcribe = boto3.client(
    "transcribe",
    region_name=AWS_REGION,
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)

dynamodb = boto3.resource(
    "dynamodb",
    region_name=AWS_REGION,
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)
table = dynamodb.Table(DDB_TABLE)


# -------------------------
# Helpers
# -------------------------
def inject_css():
    st.markdown(
        """
        <style>
          .block-container { padding-top: 1rem; }
          [data-testid="stHeader"] { height: 0rem; }
          header { visibility: hidden; }
          [data-testid="stToolbar"] { visibility: hidden; height: 0; position: fixed; }

          .vj-title { font-size: 2.0rem; font-weight: 900; margin: 0 0 .2rem 0; padding: 0; line-height: 1.15; }
          .vj-sub { opacity: .75; margin: 0 0 1.2rem 0; padding: 0; }
          .vj-card {
            border: 1px solid rgba(255,255,255,.10);
            border-radius: 18px;
            padding: 16px 18px;
            background: rgba(255,255,255,.03);
          }
          .vj-muted { opacity: .75; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def require_env():
    missing = []
    for k in [
        "OPENAI_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "TRANSCRIBE_BUCKET_NAME",
        "DYNAMODB_TABLE_NAME",
    ]:
        if not os.getenv(k):
            missing.append(k)
    if missing:
        st.error(f"Missing env vars: {', '.join(missing)}")
        st.stop()


def simple_auth():
    if not APP_PASSWORD:
        return True
    if st.session_state.get("authed"):
        return True
    pwd = st.text_input("Enter app password", type="password")
    if st.button("Login"):
        if pwd == APP_PASSWORD:
            st.session_state["authed"] = True
            st.rerun()
        else:
            st.error("Wrong password.")
    return False


def to_transcribe_media_format(ext: str) -> str:
    """
    Amazon Transcribe MediaFormat allowed values include:
    mp3, mp4, wav, flac, ogg, amr, webm, m4a
    """
    ext = (ext or "").lower().strip(".")
    if ext in ["wave"]:
        return "wav"
    if ext in ["m4a", "mp4", "mp3", "wav", "flac", "ogg", "amr", "webm"]:
        return ext
    return "wav"


def upload_bytes_to_s3(audio_bytes: bytes, ext: str, entry_date: str):
    key = f"journals/{entry_date}/{uuid.uuid4()}.{ext}"
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=audio_bytes)
    https_url = f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{key}"
    return https_url, key


def start_transcription(media_https_url: str, media_format: str) -> str:
    job_name = f"vj-{uuid.uuid4()}"
    transcribe.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={"MediaFileUri": media_https_url},
        MediaFormat=media_format,
        LanguageCode="en-US",
    )
    return job_name


def wait_for_transcription(job_name: str, poll_seconds: int = 5, timeout_seconds: int = 600):
    waited = 0
    while waited < timeout_seconds:
        job = transcribe.get_transcription_job(TranscriptionJobName=job_name)
        status = job["TranscriptionJob"]["TranscriptionJobStatus"]
        if status == "COMPLETED":
            return job["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]
        if status == "FAILED":
            return None
        time.sleep(poll_seconds)
        waited += poll_seconds
    return None


def fetch_transcript_text(transcript_uri: str) -> str:
    r = requests.get(transcript_uri, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data["results"]["transcripts"][0]["transcript"]


def save_to_dynamodb(entry_id: str, entry_date: str, media_url: str, s3_key: str, transcript: str):
    table.put_item(
        Item={
            "entry_id": entry_id,
            "entry_date": entry_date,
            "timestamp": datetime.utcnow().isoformat(),
            "audio_uri": media_url,
            "audio_key": s3_key,
            "transcription": transcript,
        }
    )


def scan_entries():
    items = []
    resp = table.scan()
    items.extend(resp.get("Items", []))
    while "LastEvaluatedKey" in resp:
        resp = table.scan(ExclusiveStartKey=resp["LastEvaluatedKey"])
        items.extend(resp.get("Items", []))
    items.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return items


def safe_aws_transcribe_from_bytes(audio_bytes: bytes, ext: str, entry_date_str: str):
    """
    Uploads bytes to S3, runs AWS Transcribe, returns transcript text or (None, err_dict).
    err_dict: {"code": "...", "message": "..."}
    """
    try:
        media_format = to_transcribe_media_format(ext)
        media_url, s3_key = upload_bytes_to_s3(audio_bytes, ext, entry_date_str)

        job_name = start_transcription(media_url, media_format)
        transcript_uri = wait_for_transcription(job_name)

        if not transcript_uri:
            return None, {"code": "TranscribeFailed", "message": "Transcription did not complete."}, (media_url, s3_key)

        txt = fetch_transcript_text(transcript_uri)
        return txt, None, (media_url, s3_key)

    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "ClientError")
        msg = e.response.get("Error", {}).get("Message", str(e))
        return None, {"code": code, "message": msg}, (None, None)
    except Exception as e:
        return None, {"code": "UnknownError", "message": str(e)}, (None, None)


# ---------- RAG ----------
def get_embedding(text: str):
    text = (text or "").strip()
    if not text:
        return None
    resp = openai.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding


def _save_index_artifacts(index, docs, meta, vectors):
    faiss.write_index(index, INDEX_FILE)
    with open(DOCS_FILE, "wb") as f:
        pickle.dump(docs, f)
    with open(META_FILE, "wb") as f:
        pickle.dump(meta, f)
    np.save(VECTORS_FILE, vectors)


def build_and_save_index(entries):
    docs, meta = [], []
    for it in entries:
        t = (it.get("transcription") or "").strip()
        if t:
            docs.append(t)
            meta.append(
                {
                    "entry_date": it.get("entry_date"),
                    "timestamp": it.get("timestamp"),
                    "entry_id": it.get("entry_id"),
                }
            )

    if not docs:
        index = faiss.IndexFlatL2(EMBED_DIM)
        vectors = np.zeros((0, EMBED_DIM), dtype="float32")
        _save_index_artifacts(index, docs, meta, vectors)
        return index, docs, meta

    embs, kept_docs, kept_meta = [], [], []
    for d, m in zip(docs, meta):
        emb = get_embedding(d)
        if emb is not None:
            embs.append(emb)
            kept_docs.append(d)
            kept_meta.append(m)

    if not embs:
        index = faiss.IndexFlatL2(EMBED_DIM)
        vectors = np.zeros((0, EMBED_DIM), dtype="float32")
        _save_index_artifacts(index, [], [], vectors)
        return index, [], []

    vectors = np.array(embs, dtype="float32")
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)

    index = faiss.IndexFlatL2(EMBED_DIM)
    index.add(vectors)
    _save_index_artifacts(index, kept_docs, kept_meta, vectors)
    return index, kept_docs, kept_meta


def load_index():
    if not (os.path.exists(INDEX_FILE) and os.path.exists(DOCS_FILE) and os.path.exists(META_FILE)):
        return None, None, None
    index = faiss.read_index(INDEX_FILE)
    with open(DOCS_FILE, "rb") as f:
        docs = pickle.load(f)
    with open(META_FILE, "rb") as f:
        meta = pickle.load(f)
    return index, docs, meta


def query_rag(question: str, index, docs, meta, k=3):
    if index is None or not docs:
        return "No transcripts indexed yet. Click **Refresh Index** after you have DynamoDB entries.", []

    q_emb = get_embedding(question)
    if q_emb is None:
        return "Your question is empty or could not be embedded.", []

    q_vec = np.array(q_emb, dtype="float32").reshape(1, -1)
    _, I = index.search(q_vec, k=min(k, len(docs)))

    hits = [{"text": docs[i], "meta": meta[i]} for i in I[0]]
    context = "\n\n".join([h["text"] for h in hits])

    prompt = f"""You are a private voice-journal assistant.
Use ONLY the journal entries below as evidence. If the answer isn't in them, say you don't know.

Journal Entries:
{context}

User Question: {question}

Answer:
"""
    resp = openai.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip(), hits


# -------------------------
# Session state init (required for discard behavior)
# -------------------------
def ensure_state():
    # Record tab state
    st.session_state.setdefault("record_audio_bytes", None)
    st.session_state.setdefault("record_audio_ext", None)
    st.session_state.setdefault("record_show_preview", False)
    st.session_state.setdefault("record_input_key", 0)

    # Voice question state
    st.session_state.setdefault("voice_q_audio_bytes", None)
    st.session_state.setdefault("voice_q_audio_ext", None)
    st.session_state.setdefault("voice_q_show_preview", False)
    st.session_state.setdefault("voice_q_transcript", "")
    st.session_state.setdefault("voice_q_input_key", 0)


ensure_state()


# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="Voice Journal", page_icon="🎙️", layout="wide")
inject_css()

# Title (HTML avoids the weird truncation you saw with st.title)
st.markdown('<div class="vj-title">🎙️ Voice Journaling Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="vj-sub">Record daily voice journals, store them, and query your memories with RAG.</div>',
    unsafe_allow_html=True,
)

require_env()
if not simple_auth():
    st.stop()

tabs = st.tabs(["🎙️ Record", "📅 Entries", "🧠 Ask (RAG)"])

# =========================================================
# TAB 1: Record
# =========================================================
with tabs[0]:
    st.markdown('<div class="vj-card">', unsafe_allow_html=True)
    st.subheader("Record today's journal")
    entry_date = st.date_input("Journal date", value=date.today())
    entry_date_str = entry_date.isoformat()
    st.caption("Record using your microphone. Preview it. Discard it. Or Save + Transcribe.")
    st.markdown("</div>", unsafe_allow_html=True)

    colA, colB = st.columns([2.2, 1])

    with colA:
        # Recorder (native streamlit)
        audio_file = st.audio_input(
            "🎤 Record journal audio",
            key=f"audio_in_record_{st.session_state.record_input_key}",
        )

        if audio_file is not None:
            audio_bytes = audio_file.read()
            st.session_state.record_audio_bytes = audio_bytes
            st.session_state.record_audio_ext = "wav"
            st.session_state.record_show_preview = True

        if st.session_state.record_show_preview and st.session_state.record_audio_bytes:
            st.audio(st.session_state.record_audio_bytes, format="audio/wav")

        st.divider()
        st.caption("Optional: upload an existing audio file instead of recording.")
        uploaded = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a", "mp4"])

        if uploaded is not None:
            st.session_state.record_audio_bytes = uploaded.read()
            st.session_state.record_audio_ext = uploaded.name.split(".")[-1].lower()
            st.session_state.record_show_preview = True
            st.audio(st.session_state.record_audio_bytes)

    with colB:
        st.markdown("#### Actions")

        discard = st.button("🗑️ Discard recording", use_container_width=True)
        save = st.button(
            "✅ Save + Transcribe",
            use_container_width=True,
            disabled=st.session_state.record_audio_bytes is None,
        )

        if discard:
            st.session_state.record_audio_bytes = None
            st.session_state.record_audio_ext = None
            st.session_state.record_show_preview = False
            st.session_state.record_input_key += 1  # remount audio_input
            st.rerun()

        if save and st.session_state.record_audio_bytes:
            chosen_bytes = st.session_state.record_audio_bytes
            chosen_ext = st.session_state.record_audio_ext or "wav"

            # Try AWS Transcribe safely (app must never crash)
            with st.spinner("Uploading + transcribing..."):
                transcript_text, err, (media_url, s3_key) = safe_aws_transcribe_from_bytes(
                    chosen_bytes, chosen_ext, entry_date_str
                )

            if err:
                st.error("Could not transcribe right now (handled safely — app will not crash).")
                st.write(f"**Error Code:** `{err['code']}`")
                st.write(f"**Message:** {err['message']}")
                st.info(
                    "If you're seeing `SubscriptionRequiredException`, you need to enable/subscribe to AWS Transcribe "
                    "on that AWS account (or use a different transcription method)."
                )
            else:
                # Save to DynamoDB only if transcription succeeded
                with st.spinner("Saving to DynamoDB..."):
                    entry_id = str(uuid.uuid4())
                    save_to_dynamodb(entry_id, entry_date_str, media_url, s3_key, transcript_text)

                st.success("Saved ✅")
                st.markdown("**Transcript:**")
                st.write(transcript_text)

                # Clear UI after save
                st.session_state.record_audio_bytes = None
                st.session_state.record_audio_ext = None
                st.session_state.record_show_preview = False
                st.session_state.record_input_key += 1
                st.rerun()

# =========================================================
# TAB 2: Entries
# =========================================================
with tabs[1]:
    st.subheader("Browse entries")
    entries = scan_entries()

    if not entries:
        st.info("No entries yet. Record one in the **Record** tab.")
    else:
        filter_date = st.date_input("Filter by date", value=date.today(), key="filter_date")
        fd = filter_date.isoformat()

        filtered = [e for e in entries if e.get("entry_date") == fd]
        st.write(f"Showing **{len(filtered)}** entry(s) for **{fd}**")

        for e in filtered:
            st.markdown(f"**{e.get('entry_date')}** — {e.get('timestamp','')}")
            st.write(e.get("transcription", ""))
            st.divider()

# =========================================================
# TAB 3: Ask (RAG)
# =========================================================
with tabs[2]:
    st.subheader("Ask your journal (RAG + LLM)")

    if "index" not in st.session_state:
        idx, docs, meta = load_index()
        st.session_state.index = idx
        st.session_state.docs = docs or []
        st.session_state.meta = meta or []

    col1, col2 = st.columns([2, 1])

    with col2:
        if st.button("🔄 Refresh Index (from DynamoDB)", use_container_width=True):
            with st.spinner("Loading DynamoDB + building FAISS..."):
                entries = scan_entries()
                idx, docs, meta = build_and_save_index(entries)
                st.session_state.index = idx
                st.session_state.docs = docs
                st.session_state.meta = meta
            st.success(f"Indexed {len(st.session_state.docs)} transcript(s).")

        st.caption(f"Indexed transcripts: {len(st.session_state.docs)}")

    with col1:
        st.markdown("#### Ask by typing")
        typed_q = st.text_input("Type a question", placeholder="e.g. What did I do last Thursday?")

        st.divider()

        st.markdown("#### Ask by voice (record → auto-transcribe → edit → ask)")
        q_audio_file = st.audio_input(
            "🎧 Record question",
            key=f"audio_in_voiceq_{st.session_state.voice_q_input_key}",
        )

        # When user records, automatically transcribe (safe, shows subscription error if any)
        if q_audio_file is not None:
            q_bytes = q_audio_file.read()
            st.session_state.voice_q_audio_bytes = q_bytes
            st.session_state.voice_q_audio_ext = "wav"
            st.session_state.voice_q_show_preview = True

            with st.spinner("Transcribing your voice question..."):
                txt, err, _ = safe_aws_transcribe_from_bytes(q_bytes, "wav", date.today().isoformat())

            if txt:
                st.session_state.voice_q_transcript = txt
            else:
                if err:
                    st.warning("Voice transcription is not available yet (AWS Transcribe not enabled/subscribed).")
                    st.write(f"**Error Code:** `{err['code']}`")
                    st.write(f"**Message:** {err['message']}")

        # Preview audio only if visible flag on
        if st.session_state.voice_q_show_preview and st.session_state.voice_q_audio_bytes:
            st.audio(st.session_state.voice_q_audio_bytes, format="audio/wav")

        # Transcript editor (always available)
        st.session_state.voice_q_transcript = st.text_area(
            "Voice transcript (edit before asking)",
            value=st.session_state.voice_q_transcript,
            height=110,
            placeholder="If transcription is enabled, your transcript will appear here automatically.",
        )

        c1, c2, c3 = st.columns([1.2, 1.4, 2.4])

        with c1:
            if st.button("🗑️ Discard voice question", use_container_width=True):
                st.session_state.voice_q_audio_bytes = None
                st.session_state.voice_q_audio_ext = None
                st.session_state.voice_q_transcript = ""
                st.session_state.voice_q_show_preview = False
                st.session_state.voice_q_input_key += 1
                st.rerun()

        with c2:
            use_voice = st.checkbox("Use voice transcript")

        with c3:
            ask_btn = st.button("Ask", use_container_width=True)

        final_q = typed_q.strip() if typed_q else ""
        if use_voice and st.session_state.voice_q_transcript.strip():
            final_q = st.session_state.voice_q_transcript.strip()

        if ask_btn and final_q:
            try:
                answer, hits = query_rag(final_q, st.session_state.index, st.session_state.docs, st.session_state.meta, k=3)
                st.markdown("### Answer")
                st.write(answer)

                if hits:
                    st.markdown("### Evidence (Top Matches)")
                    for h in hits:
                        m = h["meta"]
                        st.markdown(f"**Date:** {m.get('entry_date')}  |  **Time:** {m.get('timestamp')}")
                        st.write(h["text"])
                        st.divider()

            except Exception as e:
                st.error("RAG query failed (handled safely — app will not crash).")
                st.write(str(e))
