#!/usr/bin/env bash
set -euo pipefail

echo "🌟 Spectra AI – Unified Startup (FastAPI + React + Ollama)"
echo "========================================================="

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Helper: check command exists
need() { command -v "$1" >/dev/null 2>&1 || { echo "❌ Missing dependency: $1"; MISSING=1; }; }

MISSING=0
echo "🔍 Checking prerequisites..."
need python
need node
need npm
need ollama
if [ "$MISSING" -eq 1 ]; then
  echo "❌ Install missing prerequisites (Python 3.8+, Node.js 16+, Ollama) and re-run." >&2
  exit 1
fi
echo "✅ All core prerequisites found"

# Python venv
if [ ! -d .venv ]; then
  echo "🐍 Creating virtual environment (.venv)"
  python -m venv .venv
fi
source .venv/bin/activate

echo "📦 Ensuring latest Python dependencies"
python -m pip install --upgrade pip setuptools wheel >/dev/null
pip install --no-cache-dir -r requirements.txt >/dev/null
echo "✅ Python dependencies installed"

# Frontend deps
echo "📱 Installing/updating frontend dependencies"
pushd frontend >/dev/null
if [ ! -d node_modules ]; then
  npm install >/dev/null
else
  npm install --no-audit --no-fund >/dev/null
fi
popd >/dev/null
echo "✅ Frontend ready"

# Start Ollama if not running
if ! pgrep -f "ollama serve" >/dev/null 2>&1; then
  echo "🤖 Starting Ollama service in background"
  (nohup ollama serve >/dev/null 2>&1 &) || true
  sleep 3
else
  echo "🤖 Ollama already running"
fi

# Start FastAPI backend (gunicorn in production, uvicorn reload in dev)
BACKEND_PORT="${PORT:-5000}"
if [ "${ENVIRONMENT:-development}" = "production" ]; then
  echo "⚡ Starting FastAPI backend (gunicorn) on port $BACKEND_PORT"
  (gunicorn main:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:"$BACKEND_PORT" --workers 2 --timeout 120 >/dev/null 2>&1 &)
else
  echo "⚡ Starting FastAPI backend (uvicorn reload) on port $BACKEND_PORT"
  (python -m uvicorn main:app --host 0.0.0.0 --port "$BACKEND_PORT" --reload >/dev/null 2>&1 &)
fi

# Start React frontend
echo "⚛️ Starting React frontend (port 3000)"
(cd frontend && npm run dev >/dev/null 2>&1 &)

echo "\n✅ All services launching. Key URLs:" 
echo "   ⚡ API:           http://localhost:${PORT:-5000}" 
echo "   📚 API Docs:      http://localhost:${PORT:-5000}/docs" 
echo "   ⚛️ Frontend:      http://localhost:3000" 
echo "   🤖 Ollama API:    http://localhost:11434" 
echo "\n💜 Spectra AI is warming up emotionally and intellectually!"
