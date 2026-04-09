# NeuraChat: High-Accuracy Multi-Network RAG Architecture

NeuraChat is a state-of-the-art AI assistant designed for **Production-Grade reliability**, factual grounding, and 100% availability. It utilizes a sophisticated **6-Layer Transformer Pipeline** combined with real-time web retrieval and local SLM fallbacks.

---

## 🚀 Core Architecture: The 6-Layer Pipeline

NeuraChat does not simply "send a prompt" to an LLM. Every query passes through a rigorous multi-network verification chain:

### Network 1: Parallel Intent & Domain Classification
Upon receiving a message, the system simultaneously triggers:
- **BART Transformer**: Classifies the intent (Science, News, Jokes, Factual).
- **Parallel Search**: Initiates Google Custom Search while the intent is being detected to reduce latency.
- **Specialist Heuristics**: If the intent is "Science" or "News," the system automatically rewrites the search query with high-authority keywords (e.g., "scientific journal," "latest breaking update").

### Network 2: Multi-Query Retrieval (Recall Optimization)
To ensure we find the *correct* information:
- The system generates **3 diverse variations** of the user's question.
- It searches all 3 variations in parallel.
- Results are merged and deduplicated, casting a much wider net than standard search.

### Network 3: Adaptive Context Scoring (RAG)
Retrieved snippets are embedded using **all-MiniLM-L6-v2**. 
- **Cosine Similarity**: Comparing query embeddings against document embeddings.
- **Top-K Retrieval**: The top **10 most relevant** chunks are extracted.
- **Answerability Threshold**: If the highest relevance score is **< 0.60**, the system identifies the search as "weak" and triggers **CRAG**.

### Network 4: Corrective RAG (CRAG)
If initial search results are weak, the system:
1. Simplifies the query.
2. Performs a broader search.
3. If still no results, it informs the user rather than hallucinating.

### Network 5: Production-Grade Generation
The response is generated using **Gemini 2.0 Flash** with a strict configuration:
- **Temperature (0.1)**: Forces the model to be factual and disciplined.
- **Strict Grounding**: The system prompt explicitly forbids using prior knowledge. The model is commanded to ONLY use the provided context.

### Network 6: Accuracy Verification Pass (The Self-Critic)
Before the user sees the answer, a **second, independent pass** occurs:
- A verification prompt asks the model to fact-check the generated answer against the raw search snippets.
- If a hallucination or inaccuracy is detected, the system triggers a **Self-Correction loop** to rewrite the answer using strictly confirmed facts.

---

## 🛡️ Three-Tier Fallback Chain (Offline Reliability)

NeuraChat is designed to never fail, even if external APIs like Google are down.

1.  **Primary**: Google Gemini 2.0 Flash (Cloud-based Reasoning).
2.  **Secondary**: Hugging Face Mistral-7B (API-based Fallback).
3.  **Tertiary**: **Local SLM (Qwen-0.5B-Chat)**. 
    - This model runs entirely on your local CPU.
    - It is loaded lazily to save RAM and ensures the bot works during internet outages or API rate-limiting.

---

## 🎨 Premium User Interface (UX)

The frontend reflects the complexity of the backend through **Transparency Features**:
- **Transparency IDs**: Badges show exactly which network answered (⚡ Gemini, ⚡ HF, or ⚡ Local).
- **Reliability Indicators**: A **Verified** checkmark appears when the Network 6 Accuracy Pass successfully validates the answer.
- **Source Citation**: All retrieved sources are listed as clickable links.
- **Infinite Scroll Fix**: Specialized CSS for a contained flex-layout that allows perfectly smooth chatting.

---

## 🛠️ Technical Implementation Details

- **Backend**: FastAPI (Async for high-concurrency search tasks).
- **File Serving**: Absolute path management via `pathlib` for robust deployment.
- **Timeout Management**: 5-minute frontend timeouts to allow for heavy local CPU computation during fallbacks.
- **Launcher**: A custom `run_bot.py` script in the root that handles all environment variables and path resolution automatically.

---

## 📖 How to Run

1.  **Setup Environment**: Add your API keys (Gemini, Google, NewsData) to the `backend/.env` file.
2.  **Start the Bot**: Run `python run_bot.py` from the root directory.
3.  **Access**: Open `http://127.0.0.1:8000` in your browser.

---
**NeuraChat** — *More than a chatbot. A reliable knowledge engine.*
