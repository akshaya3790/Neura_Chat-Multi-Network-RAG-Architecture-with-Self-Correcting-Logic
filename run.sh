#!/bin/bash
echo "🚀 Starting NeuraChat AI Chatbot..."
echo ""

# Install dependencies
echo "📦 Installing Python dependencies..."
cd backend
pip install -r requirements.txt -q

echo ""
echo "⚡ Starting FastAPI server on http://localhost:8000"
echo "🌐 Open frontend/index.html in your browser"
echo ""
echo "Press Ctrl+C to stop."
echo ""

uvicorn main:app --reload --host 0.0.0.0 --port 8000
