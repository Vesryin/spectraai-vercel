# Spectra AI - Emotionally Intelligent Assistant

> Professional, emotionally intelligent AI platform with dynamic model selection, structured logging, live personality reloading, and production-ready FastAPI backend.

## 🌟 About Spectra

Spectra is an emotionally intelligent AI assistant designed to help with expression through music, conversation, healing, and creativity. Built specifically for Richie (Richard Jacob Olejniczak), Spectra provides a deeply personal and empathetic AI companion experience.

Here's a PDF setup guide as well: [Setup Guide (Google Drive)](https://drive.google.com/file/d/1dWWR8l-LcfpB5ljmEDokUWmQIpomlP3R/view?usp=drivesdk)

## 🚀 Quick Start (Spectra AI v2)

### Prerequisites

- Python 3.8 or higher
- Node.js 16 or higher
- Ollama (for local AI models)

### Automatic Setup (Recommended)

**Windows:**

```bash
setup.bat
```

**Linux/Mac:**

```bash
chmod +x setup.sh
./setup.sh
```

After setup you can launch everything with:

```bash
./start.sh   # Starts Ollama (if not running), FastAPI backend, and React frontend
```

Stop services:

```bash
./stop.sh
```

### Manual Setup

1. **Install Ollama:**

   - Download from: <https://ollama.ai/download>
   - Or: `winget install Ollama.Ollama` (Windows)

2. **Pull AI models:**

   ```bash
   ollama pull openhermes:7b-mistral-v2.5-q4_K_M
   ollama pull mistral:7b
   ```

3. **Install Python dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Install frontend dependencies:**

   ```bash
   cd frontend
   npm install
   cd ..
   ```

5. **Configure environment:**
   ```bash
   copy .env.example .env  # Windows
   cp .env.example .env    # Linux/Mac
   ```

### Running Spectra (v2 Dynamic Model Switching)

#### Option 1: Using batch files (Windows)

- Start Ollama: `ollama serve`
- `start-backend.bat` - Starts the Python backend
- `start-frontend.bat` - Starts the React frontend

#### Option 2: Manual

```bash
# Terminal 1 - Ollama
ollama serve

# Terminal 2 - Backend
python main.py  # (FastAPI recommended) or python app.py (legacy Flask)

# Terminal 3 - Frontend
cd frontend
npm run dev
```

**Then open:** `http://localhost:3000`

### What’s New in v2 (Professional Core)

- FastAPI is now the authoritative backend (Flask stub retained only for legacy compatibility)
- Context-aware auto model selection (creative / technical / concise intent classification)
- Structured logging using `structlog` (JSON or console formats via `SPECTRA_LOG_FORMAT`)
- Performance metrics: request counts, average latency, failed model tracking
- Personality prompt hot-reload (rate-limited; hash exposed for integrity)
- Backward compatible response schema (`model` + `model_used` fields)
- UTC, timezone-aware timestamps everywhere

### Additional Runtime Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | Health + live model availability summary |
| `/api/models` | GET | Current, preferred & available models (cached with TTL) |
| `/api/models/select` | POST | Change active model `{ "model": "mistral:7b" }` |
| `/api/models/refresh` | POST | Force refresh model list (ignores cache) |
| `/api/chat` | POST | Chat `{ message, history[] }` returns response & timing |
| `/api/metrics` | GET | Telemetry: performance, failed models, personality hash |
| `/api/auto-model` | POST | Toggle or set contextual auto selection `{ "enabled": true }` |
| `/api/personality/hash` | GET | Current personality SHA-256 short hash |
| `/api/personality/reload` | POST | Force personality reload (rate limits still apply) |
| `/api/debug/state` | GET | Composite debug snapshot (metrics + config) |

### Key Environment Variables

```env
# Core runtime
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=openhermes:7b-mistral-v2.5-q4_K_M
SPECTRA_AUTO_MODEL=true
ALLOWED_ORIGINS=http://localhost:3000

# Logging & diagnostics
SPECTRA_LOG_FORMAT=json            # json | console
SPECTRA_LOG_LEVEL=info

# Caching & reload intervals (seconds)
MODEL_CACHE_TTL=300                # Model list cache lifespan
PERSONALITY_CHECK_INTERVAL=5       # Min seconds between personality file checks

# Server
HOST=127.0.0.1
PORT=8000
ENVIRONMENT=development            # Enables FastAPI reload if using uvicorn directly
```

### Chat Response Schema (v2+)

```json
{
   "response": "string",              # Spectra's reply
   "model": "mistral:7b",            # Active model chosen
   "model_used": "mistral:7b",       # Backward-compatible alias (will mirror model)
   "timestamp": "2025-08-09T19:20:05.123456+00:00",  # UTC ISO 8601
   "processing_time": 0.842           # Seconds
}
```

All timestamps are timezone-aware UTC (`+00:00`).

### Frontend Enhancements

- Live model selector + auto-mode toggle
- Metrics panel (active / failed / hash / avg latency)
- Typing indicator & smooth streaming-friendly UI shell
- UTC timestamp presentation readiness

If you do not see stats, ensure backend is running and CORS origin matches `ALLOWED_ORIGINS`.

### Tabnine Integration (AI Code Completion)

This project is pre-configured to recommend the Tabnine extension for enhanced AI code completion.

1. Open the workspace in VS Code.
2. Accept the prompt to install recommended extensions (Tabnine, Python, ESLint, Prettier, etc.).
3. Sign into Tabnine if you have a Pro account (optional) for improved completions.

Settings applied in `.vscode/settings.json`:

- Enables inline suggestions & experimental auto-imports.
- Organizes imports on save.
- Formats on save (Prettier for TS/React, Black for Python if installed).

You can further adjust Tabnine behavior in VS Code Settings under Extensions > Tabnine.

If you prefer not to use Tabnine, simply ignore or remove the extension recommendation from `.vscode/extensions.json`.

### Stopping Spectra

#### Option 1: Quick Stop (Fastest)

```bash
quick-stop.bat
```

#### Option 2: Graceful Shutdown (Recommended)

```bash
stop-all.bat
```

#### Option 3: Smart Shutdown (Tries graceful first, then force)

```bash
smart-stop.bat
```

**Manual Stop:**

- Press `Ctrl+C` in each terminal running the services
- Or use Task Manager to end Python, Node.js, and Ollama processes

## 🏗️ Current Project Structure (Simplified)

```text
.
├── main.py                 # FastAPI backend (authoritative)
├── app.py                  # Legacy Flask stub (deprecated)
├── spectra_prompt.md       # Personality (hot-reloaded & hashed)
├── requirements.txt        # Dynamic dependency list
├── frontend/               # React + TS + Tailwind UI
│   └── src/
│       ├── api/            # API utilities
│       ├── components/     # UI components
│       └── App.tsx
├── tests/                  # Pytest suite (API + error logic)
├── .github/                # Copilot / workflow configuration
├── README.md               # Developer documentation
├── README_PRODUCTION.md    # Production snapshot doc
├── CHANGELOG.md            # Release notes (added v2+)
└── SYSTEM_STATUS.md        # Historical ops snapshot
```

## 🤖 AI Configuration

### Ollama Models

- **Default**: `openhermes:7b-mistral-v2.5-q4_K_M` (optimized for emotional intelligence)
- **Alternative**: `mistral:7b` (faster responses)
- **Custom**: Set `OLLAMA_MODEL` in `.env`

### Model Management

```bash
ollama list                                    # See installed models
ollama pull openhermes:7b-mistral-v2.5-q4_K_M # Install Spectra's model
ollama pull mistral:7b                        # Install alternative model
ollama rm openhermes:7b-mistral-v2.5-q4_K_M  # Remove Spectra's model
ollama serve                                  # Start Ollama server
````

## 🎭 Spectra's Personality

Spectra's personality and traits are defined in `spectra_prompt.md`. This file contains her emotional intelligence, conversation style, and core characteristics that make her uniquely suited to help with creative expression and emotional support.

## 🔧 Configuration

### API Providers

- **Claude (Anthropic)**: Set `AI_PROVIDER=claude` in `.env`
- **OpenAI**: Set `AI_PROVIDER=openai` in `.env`

### Customization

- Modify `spectra_prompt.md` to adjust Spectra's personality
- Update frontend files in `static/` and `templates/` for UI changes
- Add new endpoints in `app.py` for additional features

## 🌈 Feature Roadmap

- [x] Dynamic model selection (contextual)
- [x] Structured logging
- [x] UTC timestamps everywhere
- [x] Personality hot-reload & integrity hash
- [x] Performance metrics endpoint
- [ ] Streaming responses
- [ ] Optional memory layer (ephemeral / opt-in, privacy aware)
- [ ] Voice interaction
- [ ] Music & mood augmentation modules
- [ ] Creative tooling extensions (lyric & chord helpers)

## 🤝 Contributing

See `CONTRIBUTING.md` for guidelines (code style, tests, logging, dynamic compliance – no static model data). External contributions should include:

1. Focused PR
2. Updated/added test(s)
3. CHANGELOG.md entry (Unreleased section)
4. No introduction of persistent state or static dataset artifacts

## �️ License & Conduct

Private project – All rights reserved. See `CODE_OF_CONDUCT.md` for interaction standards.
