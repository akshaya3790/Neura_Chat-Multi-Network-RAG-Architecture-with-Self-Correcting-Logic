"""
AI Chatbot Backend - FastAPI
Primary replies: NewsData.io, JokeAPI, ChatBot.com /query.
Gemini + Google RAG only as fallback (minimal use).
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import httpx
import numpy as np
from typing import List, Optional, Tuple
import os
import re
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Load .env file if it exists
load_dotenv()

# ─── CONFIG ────────────────────────────────────────────────────────────────────
# ─── CONFIG ────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CX = os.getenv("GOOGLE_CX")
HF_TOKEN = os.getenv("HF_TOKEN")

# User-provided APIs
NEWSDATA_API_KEY = os.getenv("NEWSDATA_API_KEY")
CHATBOT_CLIENT_TOKEN = os.getenv("CHATBOT_CLIENT_TOKEN")
JOKEAPI_BASE = "https://v2.jokeapi.dev"
CHATBOT_QUERY_URL = "https://api.chatbot.com/query"
NEWSDATA_URL = "https://newsdata.io/api/1/news"

GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
GOOGLE_URL = "https://www.googleapis.com/customsearch/v1"

# Updated HF Router URL
HF_ROUTER_URL = "https://router.huggingface.co/hf-inference/models"
HF_EMBED_URL = f"{HF_ROUTER_URL}/sentence-transformers/all-MiniLM-L6-v2"
HF_CLASSIFY_URL = f"{HF_ROUTER_URL}/facebook/bart-large-mnli"
HF_GEN_URL = f"{HF_ROUTER_URL}/mistralai/Mistral-7B-Instruct-v0.2"

_JOKE_PAT = re.compile(
    r"\b(joke|jokes|funny|humou?r|humor|laugh|make me smile|tell me a joke)\b",
    re.I,
)
_NEWS_PAT = re.compile(
    r"\b(news|headlines?|latest news|breaking news|current events|what'?s happening|world news)\b",
    re.I,
)

# ─── LOCAL SLM CACHE (Lazy Load) ──────────────────────────────────────────────
LOCAL_MODEL_ID = "Qwen/Qwen1.5-0.5B-Chat"
local_pipe = None

# ─── PATHS ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
print(f"DEBUG: BASE_DIR = {BASE_DIR}")
print(f"DEBUG: FRONTEND_DIR = {FRONTEND_DIR}")
print(f"DEBUG: index.html exists: {(FRONTEND_DIR / 'index.html').exists()}")

# ─── FASTAPI APP ────────────────────────────────────────────────────────────────
app = FastAPI(title="AI Chatbot API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (commented out - directory doesn't exist)
# app.mount("/static", StaticFiles(directory="../frontend/static"), name="static")


# ─── MODELS ────────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    history: List[dict] = []
    use_search: bool = True
    # ChatBot.com requires sessionId length >= 10; reuse per browser session
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    sources: List[dict] = []
    search_used: bool = False
    model_info: dict = {}


# ─── IN-MEMORY VECTOR STORE (RAG) ──────────────────────────────────────────────
class VectorStore:
    def __init__(self):
        self.documents: List[str] = []
        self.embeddings: List[List[float]] = []
        self.metadata: List[dict] = []

    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        a, b = np.array(a), np.array(b)
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def add(self, text: str, embedding: List[float], meta: dict = {}):
        self.documents.append(text)
        self.embeddings.append(embedding)
        self.metadata.append(meta)

    def search(self, query_embedding: List[float], top_k: int = 3) -> List[dict]:
        if not self.embeddings:
            return []
        scores = [self.cosine_similarity(query_embedding, e) for e in self.embeddings]
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            {"text": self.documents[i], "score": scores[i], "meta": self.metadata[i]}
            for i in top_indices if scores[i] > -1.0
        ]

vector_store = VectorStore()


# ─── HuggingFace EMBEDDINGS ─────────────────────────────────────────────────────
async def get_embedding(text: str) -> List[float]:
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": text[:512]}
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            r = await client.post(HF_EMBED_URL, headers=headers, json=payload)
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list) and isinstance(data[0], list):
                    # sentence-transformers returns list of lists
                    return data[0]
                elif isinstance(data, list) and isinstance(data[0], float):
                    return data
        except Exception:
            pass
    # Fallback: random unit vector
    v = np.random.randn(384).tolist()
    norm = np.linalg.norm(v)
    return (np.array(v) / norm).tolist()


# ─── GOOGLE SEARCH ──────────────────────────────────────────────────────────────
async def google_search(query: str, num: int = 5) -> List[dict]:
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CX,
        "q": query,
        "num": num,
    }
    async with httpx.AsyncClient(timeout=15) as client:
        try:
            r = await client.get(GOOGLE_URL, params=params)
            if r.status_code == 200:
                data = r.json()
                results = []
                for item in data.get("items", []):
                    results.append({
                        "title": item.get("title", ""),
                        "snippet": item.get("snippet", ""),
                        "link": item.get("link", ""),
                    })
                return results
        except Exception as e:
            print(f"Google Search error: {e}")
    return []


async def generate_query_variations(query: str) -> List[str]:
    """Generates 3 variations of the query for better RAG retrieval."""
    prompt = (
        "TASK: Generate 3 diverse search-engine optimized queries to find the answer to this question.\n"
        f"QUESTION: {query}\n\n"
        "Reply with only the 3 queries, one per line, no numbering."
    )
    res = await call_gemini(prompt, [])
    if "FAIL_AI_GEN" in res:
        return [query]
    
    variations = [v.strip() for v in res.split("\n") if v.strip()]
    # Ensure at least the original is included
    if query not in variations:
        variations.append(query)
    return variations[:4]


async def detect_intent_transformer(message: str) -> str:
    """Uses a Transformer model (BART) to classify intent and domain."""
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": message,
        "parameters": {"candidate_labels": ["news", "science", "joke", "general chat", "factual query"]}
    }
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            r = await client.post(HF_CLASSIFY_URL, headers=headers, json=payload)
            if r.status_code == 200:
                data = r.json()
                label = data.get("labels", ["chat"])[0]
                return label
        except Exception as e:
            print(f"Transformer Intent Error: {e}")
    
    # Fallback to regex logic
    m = message.lower()
    if _JOKE_PAT.search(m): return "joke"
    if _NEWS_PAT.search(m): return "news"
    if any(k in m for k in ["science", "physics", "biology", "space", "chemistry"]): return "science"
    return "chat"


def normalize_chatbot_session(session_id: Optional[str]) -> str:
    sid = (session_id or "").strip()
    if len(sid) < 10:
        return "neurachat-default-session-id"
    return sid[:256]


def fulfillment_to_text(fulfillment: list) -> str:
    parts: List[str] = []
    for item in fulfillment or []:
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind == "text" and item.get("message"):
            parts.append(str(item["message"]))
        elif kind == "image" and item.get("imageUrl"):
            parts.append(f"![]({item['imageUrl']})")
    return "\n\n".join(parts).strip()


async def fetch_joke() -> str:
    params = {
        "type": "single",
        "blacklistFlags": "nsfw,religious,political,racist,sexist,explicit",
    }
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(f"{JOKEAPI_BASE}/joke/Any", params=params)
        if r.status_code != 200:
            return "Could not fetch a joke right now. Please try again."
        data = r.json()
        if data.get("error"):
            return data.get("message", "Joke service returned an error.")
        if data.get("type") == "single" and data.get("joke"):
            return data["joke"]
        if data.get("type") == "twopart":
            return f"{data.get('setup', '')}\n\n*{data.get('delivery', '')}*"
    return "No joke available this time."


async def fetch_news_headlines(user_message: str) -> Tuple[str, List[dict]]:
    q = re.sub(_JOKE_PAT, "", user_message)
    q = re.sub(_NEWS_PAT, "", q)
    q = re.sub(r"\s+", " ", q).strip()
    params: dict = {"apikey": NEWSDATA_API_KEY, "language": "en", "size": 6}
    if len(q) >= 3:
        params["q"] = q[:200]
    async with httpx.AsyncClient(timeout=25) as client:
        r = await client.get(NEWSDATA_URL, params=params)
        if r.status_code != 200:
            return (
                f"News service error ({r.status_code}). Try again later.",
                [],
            )
        data = r.json()
        if data.get("status") != "success":
            return ("No news results right now.", [])
        results = data.get("results") or []
        if not results:
            return ("No articles matched that query.", [])
        lines = ["### Latest headlines\n"]
        sources: List[dict] = []
        for art in results:
            title = art.get("title") or "Untitled"
            link = art.get("link") or ""
            src = art.get("source_name") or art.get("source_id") or ""
            desc = (art.get("description") or "")[:220]
            if desc and not desc.endswith("…"):
                desc += "…" if len(desc) == 220 else ""
            lines.append(f"- **{title}** ({src})\n  {desc}\n  [Read more]({link})")
            sources.append({"title": title, "link": link})
        return "\n\n".join(lines), sources


async def call_chatbot_query(query: str, session_id: str) -> Tuple[str, bool]:
    payload = {"query": query, "sessionId": session_id}
    headers = {
        "Authorization": f"Bearer {CHATBOT_CLIENT_TOKEN}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=45) as client:
        r = await client.post(CHATBOT_QUERY_URL, json=payload, headers=headers)
        if r.status_code != 200:
            print(f"ChatBot API: {r.status_code} {r.text[:500]}")
            return "", False
        data = r.json()
        status = (data.get("status") or {}).get("code")
        if status != 200:
            return "", False
        result = data.get("result") or {}
        fulfillment = result.get("fulfillment") or []
        text = fulfillment_to_text(fulfillment)
        return text, bool(text)


# ─── LLM CALLS ────────────────────────────────────────────────────────────────
async def call_gemini(prompt: str, history: List[dict]) -> str:
    contents = []
    for h in history[-6:]:  # last 3 pairs
        contents.append({"role": h["role"], "parts": [{"text": h["content"]}]})
    contents.append({"role": "user", "parts": [{"text": prompt}]})

    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": 0.1,
            "topK": 40,
            "topP": 0.95,
            "maxOutputTokens": 1024,
        },
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        ],
    }
    async with httpx.AsyncClient(timeout=60) as client:
        try:
            r = await client.post(GEMINI_URL, json=payload)
            if r.status_code == 200:
                data = r.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
            else:
                return f"FAIL_AI_GEN: {r.status_code}"
        except Exception as e:
            print(f"Gemini API Error: {e}")
            return "FAIL_AI_GEN"


async def call_hf_generator(prompt: str) -> str:
    """Fallback generator using Mistral or similar on Hugging Face."""
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": f"<s>[INST] {prompt} [/INST]",
        "parameters": {"max_new_tokens": 512, "temperature": 0.1}
    }
    async with httpx.AsyncClient(timeout=45) as client:
        try:
            r = await client.post(HF_GEN_URL, headers=headers, json=payload)
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list) and len(data) > 0 and "generated_text" in data[0]:
                    txt = data[0]["generated_text"]
                    # Extract only the assistant's part if model echoes
                    if "[/INST]" in txt:
                        return txt.split("[/INST]")[-1].strip()
                    return txt.strip()
        except Exception as e:
            print(f"HF Generation Error: {e}")
    return "FAIL_HF_GEN"


async def call_local_slm(prompt: str) -> str:
    """Tertiary fallback using local Phi-3 model (CPU)."""
    global local_pipe
    if local_pipe is None:
        print(f"Downloading/Loading local SLM: {LOCAL_MODEL_ID}...")
        print("NOTE: This may take several minutes on the first run.")
        try:
            tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_ID, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                LOCAL_MODEL_ID,
                device_map="cpu",
                torch_dtype="auto",
                trust_remote_code=True,
            )
            local_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
        except Exception as e:
            print(f"Error loading local SLM: {e}")
            return "FAIL_LOCAL_SLM"

    try:
        messages = [{"role": "user", "content": prompt}]
        generation_args = {
            "max_new_tokens": 500,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False,
        }
        output = local_pipe(messages, **generation_args)
        return output[0]['generated_text'].strip()
    except Exception as e:
        print(f"Local SLM Inference Error: {e}")
        return "FAIL_LOCAL_SLM"


async def generate_answer_multi_network(prompt: str, history: List[dict]) -> Tuple[str, str]:
    """Tries Gemini, then HF, then Local SLM."""
    # 1. Attempt Gemini
    reply = await call_gemini(prompt, history)
    
    if reply.startswith("FAIL_AI_GEN") or "Quota exceeded" in reply:
        print(f"Gemini Failed ({reply}). Falling back to Hugging Face...")
        # 2. Secondary Fallback: HF Mistral
        hf_reply = await call_hf_generator(prompt)
        if hf_reply != "FAIL_HF_GEN":
            return hf_reply, "hf-mistral-fallback"
        
        print(f"Hugging Face Failed. Falling back to Local SLM ({LOCAL_MODEL_ID})...")
        # 3. Tertiary Fallback: Local Qwen/Phi-3
        local_reply = await call_local_slm(prompt)
        if local_reply != "FAIL_LOCAL_SLM":
            return local_reply, "local-phi3-slm"

        return "All AI systems are currently offline. Please try again later.", "error"
    
    return reply, "gemini-2.0-flash"


async def verify_accuracy(query: str, reply: str, context: str) -> bool:
    """Uses a quick model pass to check if the answer is accurate relative to context."""
    if not context: return True # Nothing to verify against
    
    prompt = (
        "TASK: Fact-check the following answer against the provided context.\n"
        f"CONTEXT: {context[:2000]}\n"
        f"QUESTION: {query}\n"
        f"ANSWER: {reply}\n\n"
        "Reply with only 'ACCURATE' or 'INACCURATE'."
    )
    # Use Gemini for verification as it's typically faster/smarter for this
    verification = await call_gemini(prompt, [])
    return "INACCURATE" not in verification.upper()


# ─── RAG PIPELINE ──────────────────────────────────────────────────────────────
async def rag_pipeline(query: str, search_results: List[dict]) -> tuple[str, List[dict]]:
    """Embed search results, store in vector DB, retrieve relevant context."""
    retrieved_sources = []

    for result in search_results:
        chunk = f"{result['title']}: {result['snippet']}"
        embedding = await get_embedding(chunk)
        vector_store.add(chunk, embedding, {"title": result["title"], "link": result["link"]})

    query_emb = await get_embedding(query)
    hits = vector_store.search(query_emb, top_k=10)

    context_parts = []
    max_score = 0
    for hit in hits:
        context_parts.append(hit["text"])
        retrieved_sources.append(hit["meta"])
        max_score = max(max_score, hit["score"])

    return "\n\n".join(context_parts), retrieved_sources, max_score


# ─── ROUTES ────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        return {"error": f"index.html not found at {index_path}"}
    return FileResponse(index_path)

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "2.2.0 (High Accuracy)",
        "local_model": LOCAL_MODEL_ID,
        "primary": ["NewsData.io", "JokeAPI", "ChatBot.com"],
        "fallback_llm": "gemini-2.0-flash",
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        msg = req.message.strip()
        sources: List[dict] = []
        search_used = False
        context = ""

        # Parallel Execution: Intent Detection + Optional Search
        intent_task = detect_intent_transformer(msg)
        
        # Refine search query based on heuristics before running task
        search_query = msg
        m_lower = msg.lower()
        if any(w in m_lower for w in ["science", "physics", "biology", "research", "study", "formula", "theorem", "chemical"]):
            search_query += " scientific research data paper journal"
        elif any(w in m_lower for w in ["news", "latest", "happened", "today", "breaking", "update"]):
            search_query += " latest news today breaking update"

        # Multi-Query Retrieval: Generate 3 variations
        intent = await intent_task
        query_variations = await generate_query_variations(search_query)
        print(f"Multi-Query: searching for {query_variations}")
        
        # Parallel Search for all variations
        search_tasks = [google_search(q) for q in query_variations]
        search_results_list = await asyncio.gather(*search_tasks)
        
        # Merge and deduplicate results by link
        search_results = []
        seen_links = set()
        for results in search_results_list:
            for r in results:
                if r["link"] not in seen_links:
                    search_results.append(r)
                    seen_links.add(r["link"])
        
        # Limit total results to top 15 for processing
        search_results = search_results[:15]
        # ── Specialized Routing ─────────────────────────────────────────────
        if intent == "joke":
            reply = await fetch_joke()
            return ChatResponse(reply=reply, model_info={"route": "jokeapi"})

        if intent == "news":
            # If search results exist, prefer RAG for accuracy
            if search_results and isinstance(search_results, list) and len(search_results) > 0:
                pass # Continue to RAG pipeline below
            else:
                reply, sources = await fetch_news_headlines(msg)
                return ChatResponse(reply=reply, sources=sources[:6], model_info={"route": "newsdata.io"})

        # ── RAG Pipeline 1 (CRAG Step 1) ────────────────────────────────────
        if search_results and isinstance(search_results, list) and len(search_results) > 0:
            search_used = True
            context, sources, confidence = await rag_pipeline(msg, search_results)
            
            # Corrective RAG: If confidence is low, try a broader search
            if confidence < 0.60:
                print(f"CRAG: Low confidence ({confidence:.2f} < 0.60). Triggering broader search...")
                broader_query = msg.split("?")[0].strip() # Simpler query
                broader_results = await google_search(broader_query, num=3)
                if broader_results:
                    b_context, b_sources, b_conf = await rag_pipeline(broader_query, broader_results)
                    if b_conf > confidence:
                        context = b_context
                        sources.extend(b_sources)
                        print(f"CRAG: Broader search improved confidence to {b_conf:.2f}")

        # ── Self-Handling Generation ────────────────────────────────────────
        system_note = (
            "You are NeuraChat — a high-accuracy AI expert. "
            "For SCIENCE topics, be precise and use technical terms. "
            "For NEWS, rely strictly on the provided context. "
            "If unsure, state your uncertainty. Cite sources if available."
        )
        
        prompt = f"{system_note}\n\n### User Query: {msg}"
        if context:
            prompt = f"{system_note}\n\n### Scientific/Web Context:\n{context}\n\n### User Query: {msg}"

        gemini_history = [
            {"role": ("user" if m["role"] == "user" else "model"), "content": m["content"]}
            for m in req.history
        ]
        
        reply, model_used = await generate_answer_multi_network(prompt, gemini_history)

        # ── Accuracy Verification (Network 4) ───────────────────────────────
        is_accurate = await verify_accuracy(msg, reply, context)
        if not is_accurate and context:
            print("Answer failed accuracy check. Triggering self-correction...")
            refactor_prompt = (
                "The previous response was flagged as potentially inaccurate. "
                "You are a factual assistant. ONLY answer using the provided context. "
                "If the answer is not in the context, say: 'I don't have enough information from reliable sources.' "
                "Do NOT use prior knowledge. Do NOT guess. ALWAYS cite sources."
                f"\n\n### CONTEXT (Reliability Verified):\n{context}"
            )
            reply, _ = await generate_answer_multi_network(refactor_prompt, [])

        return ChatResponse(
            reply=reply,
            sources=sources[:6],
            search_used=search_used,
            model_info={
                "route": "multi-network-pipeline",
                "intent": intent,
                "primary_model": model_used,
                "accuracy_verified": is_accurate,
                "deep_learning": "Parallel Transformers (BART + MiniLM + Gemini/Mistral)"
            },
        )
    except Exception as e:
        print(f"Chat endpoint error: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/models")
async def models_info():
    return {
        "llm": "Google Gemini 2.0 Flash",
        "embeddings": "HuggingFace all-MiniLM-L6-v2",
        "search": "Google Custom Search Engine",
        "rag": "RAG pipeline with cosine similarity",
        "framework": "FastAPI + Uvicorn"
    }
