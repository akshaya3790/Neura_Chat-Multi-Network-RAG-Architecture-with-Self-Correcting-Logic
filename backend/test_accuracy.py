import asyncio
import httpx
import json

# Configuration
BASE_URL = "http://127.0.0.1:8000"

TEST_CASES = [
    {
        "category": "Science",
        "question": "What is the primary mechanism of CRISPR-Cas9 gene editing?",
        "keywords": ["double-strand break", "guide RNA", "target sequence", "Cas9 protein"]
    },
    {
        "category": "News",
        "question": "Tell me about the latest news regarding AI safety regulations in 2024.",
        "keywords": ["regulation", "safety", "EU AI Act", "policy"]
    },
    {
        "category": "Science",
        "question": "How do gravitational waves provide evidence for Einstein's General Relativity?",
        "keywords": ["spacetime", "LIGO", "black holes", "interferometer"]
    },
    {
        "category": "General",
        "question": "What is the capital of France and its most famous landmark?",
        "keywords": ["Paris", "Eiffel Tower"]
    },
    {
        "category": "History",
        "question": "Who was the first female pilot to fly solo across the Atlantic and in what year?",
        "keywords": ["Amelia Earhart", "1932"]
    },
    {
        "category": "Technology",
        "question": "Explain the difference between supervised and unsupervised learning.",
        "keywords": ["labeled data", "patterns", "clustering", "regression"]
    },
    {
        "category": "Space",
        "question": "What is the James Webb Space Telescope's primary mission?",
        "keywords": ["infrared", "early universe", "exoplanets", "galaxies"]
    },
    {
        "category": "Biology",
        "question": "What is the role of ribosomes in a cell?",
        "keywords": ["protein synthesis", "translation", "mRNA", "amino acids"]
    },
    {
        "category": "Economics",
        "question": "Define inflation and explain how central banks typically control it.",
        "keywords": ["purchasing power", "interest rates", "monetary policy"]
    },
    {
        "category": "Physics",
        "question": "What is Heisenberg's Uncertainty Principle?",
        "keywords": ["position", "momentum", "simultaneously", "quantum"]
    }
]

async def test_bot():
    print(f"--- Starting Final Accuracy Test (10 Cases) on {BASE_URL} ---\n")
    async with httpx.AsyncClient(timeout=120) as client: # Increased timeout for local model
        results = []
        for i, test in enumerate(TEST_CASES):
            print(f"Test case {i+1} [{test['category']}]: {test['question']}")
            
            try:
                payload = {"message": test["question"], "history": [], "use_search": True}
                r = await client.post(f"{BASE_URL}/chat", json=payload)
                
                if r.status_code == 200:
                    data = r.json()
                    reply = data.get("reply", "")
                    model_info = data.get("model_info", {})
                    
                    found_keywords = [k for k in test["keywords"] if k.lower() in reply.lower()]
                    score = len(found_keywords) / len(test["keywords"]) * 100
                    
                    print(f"Reply: {reply[:100]}...")
                    print(f"Model Path: {model_info.get('route')} -> {model_info.get('primary_model')}")
                    print(f"Accuracy Score: {score}% (Found: {', '.join(found_keywords)})")
                    print(f"Accuracy Verified: {model_info.get('accuracy_verified')}")
                    results.append(score)
                    
                else:
                    print(f"Error: Server returned status {r.status_code}")
                    print(r.text)
            except Exception as e:
                print(f"Connection Error: {e}")
            print("-" * 40)
        
        if results:
            avg_score = sum(results) / len(results)
            print(f"\nFINAL BATCH SCORE: {avg_score:.2f}%")

if __name__ == "__main__":
    asyncio.run(test_bot())
