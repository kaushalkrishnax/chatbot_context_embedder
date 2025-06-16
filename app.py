import gradio as gr
import aiohttp
import asyncio
import json
import hashlib
import re
import logging
from typing import Any, List, Dict, Optional
from functools import lru_cache

# --- Configuration ---
REQUESTS_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}
MIN_TEXT_LENGTH = 10
CHUNK_TARGET_SIZE = 150
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 32
STOPWORDS = {'the', 'and', 'for', 'with', 'this', 'that'}

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Lazy Load Model ---
@lru_cache(maxsize=1)
def get_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL)

# --- Utility Functions ---
@lru_cache(maxsize=1000)
def get_hash(text: str, title: str, source: str) -> str:
    return hashlib.md5((text + title + source).encode()).hexdigest()

@lru_cache(maxsize=1000)
def extract_keywords(text: str, top_k: int = 5) -> tuple:
    words = re.findall(r'\b\w+\b', text.lower())
    word_freq = {}
    for word in words:
        if len(word) > 3 and word not in STOPWORDS:
            word_freq[word] = word_freq.get(word, 0) + 1
    return tuple(sorted(word_freq, key=word_freq.get, reverse=True)[:top_k])

def generate_title(text: str, fallback: str = "Untitled") -> str:
    keywords = extract_keywords(text, top_k=3)
    if keywords:
        return " ".join(keywords[:3]).title()[:30]
    return fallback

def sent_tokenize_fallback(text: str) -> List[str]:
    return re.split(r'(?<=[.!?])\s+', text.strip())

def chunk_text(text: str, fallback_title: str = "Untitled", chunk_size: int = CHUNK_TARGET_SIZE) -> List[Dict[str, str]]:
    if len(text.strip()) < MIN_TEXT_LENGTH:
        return []

    sentences = sent_tokenize_fallback(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_words = len(sentence.split())
        if current_length + sentence_words > chunk_size and current_chunk:
            chunk_body = " ".join(current_chunk)
            chunks.append({"title": generate_title(chunk_body, fallback_title), "text": chunk_body})
            current_chunk = []
            current_length = 0
        current_chunk.append(sentence)
        current_length += sentence_words

    if current_chunk:
        chunk_body = " ".join(current_chunk)
        chunks.append({"title": generate_title(chunk_body, fallback_title), "text": chunk_body})

    return chunks

async def extract_from_url(url: str, session: aiohttp.ClientSession) -> Optional[Dict[str, str]]:
    try:
        async with session.get(url, timeout=10, headers=REQUESTS_HEADERS) as response:
            content = await response.read()
            encoding = response.charset or 'utf-8'
            html = content.decode(encoding, errors='ignore')

            import trafilatura
            metadata_doc = trafilatura.extract_metadata(html)
            fallback_title = metadata_doc.as_dict().get('title', 'Untitled') if metadata_doc else 'Untitled'
            text = trafilatura.extract(html)

            if not text:
                logger.warning("Trafilatura failed for %s, using BeautifulSoup fallback...", url)
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html, "html.parser")
                for tag in soup(["script", "style", "nav", "footer", "noscript"]):
                    tag.decompose()
                text = soup.get_text(" ", strip=True)

            return {"title": fallback_title, "text": text} if text and len(text.strip()) > MIN_TEXT_LENGTH else None

    except Exception as e:
        logger.error("Failed to process URL %s: %s", url, e)
        return None

# --- Vector Store ---
class VectorStore:
    def __init__(self):
        self.dedup_hashes = set()
        self.results = []
        self._chunks_to_embed = []

    def reset(self):
        self.dedup_hashes.clear()
        self.results.clear()
        self._chunks_to_embed.clear()

    async def add_chunks(self, text: str, source: str, title: str, extra_payload: Optional[Dict] = None):
        if len(text.strip()) < MIN_TEXT_LENGTH:
            return

        for chunk in chunk_text(text, fallback_title=title):
            chunk_id = get_hash(chunk['text'], chunk['title'], source)
            if chunk_id in self.dedup_hashes:
                continue
            self.dedup_hashes.add(chunk_id)
            payload = {
                "title": chunk['title'],
                "text": chunk['text'],
                "source": source,
                **(extra_payload or {})
            }
            self._chunks_to_embed.append({
                "id": chunk_id,
                "text": chunk['text'],
                "payload": payload
            })

    def embed_and_finalize(self, batch_size: int = EMBEDDING_BATCH_SIZE):
        if not self._chunks_to_embed:
            return

        all_texts = [item['text'] for item in self._chunks_to_embed]
        logger.info(f"Starting batch embedding for {len(all_texts)} chunks...")
        all_vectors = get_model().encode(all_texts, batch_size=batch_size, show_progress_bar=False)
        logger.info("Batch embedding complete.")

        seen_hashes = set()
        for item, vector in zip(self._chunks_to_embed, all_vectors):
            if item['id'] not in seen_hashes:
                seen_hashes.add(item['id'])
                self.results.append({
                    "id": item['id'],
                    "vector": vector.tolist(),
                    "payload": item['payload']
                })

# --- Processing Logic ---
async def traverse_and_process(data: Any, vector_store: VectorStore, source_key: str = "json_root", session: aiohttp.ClientSession = None, processed_urls: set = None):
    if processed_urls is None:
        processed_urls = set()

    if isinstance(data, dict):
        for key, value in data.items():
            await traverse_and_process(value, vector_store, source_key=key, session=session, processed_urls=processed_urls)
    elif isinstance(data, list):
        for item in data:
            await traverse_and_process(item, vector_store, source_key=source_key, session=session, processed_urls=processed_urls)
    elif isinstance(data, str) and data.strip():
        content = data.strip()
        if content.startswith("http") and content not in processed_urls:
            processed_urls.add(content)
            extracted = await extract_from_url(content, session)
            if extracted:
                await vector_store.add_chunks(extracted['text'], source=content, title=extracted['title'])
        else:
            await vector_store.add_chunks(content, source="json_text", title=generate_title(content, source_key))

async def process_input_async(input_json: str) -> str:
    vector_store = VectorStore()
    try:
        input_obj = json.loads(input_json)
        if not isinstance(input_obj, list):
            return json.dumps({"error": "Input must be a list of {type, content} items"}, indent=2)

        async with aiohttp.ClientSession() as session:
            tasks = []
            processed_urls = set()
            for entry in input_obj:
                kind = entry.get("type")
                content = entry.get("content")
                if not kind or not content:
                    continue

                if kind == "context":
                    tasks.append(traverse_and_process(content, vector_store, session=session, processed_urls=processed_urls))
                elif kind == "text":
                    keywords = extract_keywords(content)
                    tasks.append(vector_store.add_chunks(content, source="user_message", title=generate_title(content, "User Message"), extra_payload={"keywords": list(keywords)}))

            await asyncio.gather(*tasks)

        vector_store.embed_and_finalize()
        return json.dumps(vector_store.results, indent=2)
    except json.JSONDecodeError as e:
        return json.dumps({"error": "Invalid JSON provided", "details": str(e)}, indent=2)
    except Exception as e:
        logger.error("Unexpected error during processing: %s", e, exc_info=True)
        return json.dumps({"error": "An unexpected error occurred", "details": str(e)}, indent=2)

def process_input(input_json: str) -> str:
    return asyncio.run(process_input_async(input_json))

# --- Gradio UI ---
iface = gr.Interface(
    fn=process_input,
    inputs=gr.Textbox(
        label="Paste JSON Array [{type, content}]",
        lines=15,
        placeholder='[{"type": "context", "content": {"url": "https://www.gradio.app/guides/quickstart", "summary": "Sample summary..."}}, {"type": "text", "content": "What is the main topic of this page?"}]'
    ),
    outputs=gr.Code(label="Qdrant-Ready Embeddings", language="json"),
    title="Fast Messaging Embedder",
    description="Optimized for messaging: processes text and URLs, generates descriptive titles, and produces embeddings for Qdrant.",
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch()
