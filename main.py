import os
import time
import hashlib
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from pydantic import BaseModel

# Assume ChatMessage, StatusResponse, ModelListResponse, ModelSelectRequest, ModelSelectResponse,
# ChatRequest, ChatResponse, ToggleAutoModelRequest, and your provider interfaces are defined elsewhere.

import logging

logger = logging.getLogger("spectra")
logger.setLevel(logging.INFO)

class SpectraAI:
    def __init__(self):
        self.model = os.getenv('DEFAULT_MODEL', 'ollama:openhermes')
        self.preferred_model = self.model
        self.available_models: List[str] = []
        self.available_providers: List[str] = []
        self.providers = self._init_providers()
        self.failed_models = set()
        self.personality_prompt = "You are Spectra, a calm, wise, and emotionally intelligent AI."
        self.personality_hash = ""
        self._personality_path = self._resolve_personality_path()
        self._personality_mtime: Optional[float] = None
        self.auto_model_enabled = True
        self.request_count = 0
        self.total_processing_time = 0.0
        self.model_cache_ttl = 300  # seconds
        self.current_provider = "ollama"

        self.refresh_models()
        self._maybe_reload_personality()

    def _init_providers(self) -> Dict[str, Any]:
        # Initialize AI providers: Ollama, OpenAI, Anthropic, etc.
        # For example purposes, assume these are classes that wrap provider APIs
        providers = {}
        # Pseudocode:
        # if Ollama API key or socket available:
        #     providers['ollama'] = OllamaProvider()
        # if OpenAI API key present:
        #     providers['openai'] = OpenAIProvider()
        # if Anthropic API key present:
        #     providers['anthropic'] = AnthropicProvider()
        return providers

    def _resolve_personality_path(self) -> Optional[str]:
        path = os.getenv('PERSONALITY_PATH', 'personality.txt')
        if os.path.exists(path):
            return path
        else:
            logger.warning(f"Personality file not found at {path}")
            return None

    def refresh_models(self):
        # Refresh available models from each provider
        all_models = []
        available_providers = []
        for provider_name, provider in self.providers.items():
            try:
                models = provider.get_models()
                all_models.extend([f"{provider_name}:{m}" for m in models])
                available_providers.append(provider_name)
            except Exception as e:
                logger.warning(f"Failed to refresh models for {provider_name}: {e}")

        self.available_models = all_models
        self.available_providers = available_providers

        # Validate current model
        if self.model not in self.available_models:
            if self.available_models:
                self.model = self.available_models[0]
            else:
                self.model = None

    def _maybe_reload_personality(self):
        try:
            if not self._personality_path:
                return

            current_mtime = os.path.getmtime(self._personality_path)
            if self._personality_mtime and current_mtime <= self._personality_mtime:
                return

            with open(self._personality_path, encoding='utf-8') as f:
                content = f.read()

            new_hash = self._hash_personality(content)
            if new_hash != self.personality_hash:
                self.personality_prompt = content.strip()
                self.personality_hash = new_hash
                self._personality_mtime = current_mtime
                logger.info("personality_reloaded", hash=self.personality_hash)

        except Exception as e:
            logger.warning("personality_reload_failed", error=str(e))
current_mtime = self._personality_path.stat().st_mtime
            if self._personality_mtime and current_mtime <= self._personality_mtime:
                return

            content = self._personality_path.read_text(encoding='utf-8')
            new_hash = self._hash_personality(content)

            if new_hash != self.personality_hash:
                self.personality_prompt = content.strip()
                self.personality_hash = new_hash
                self._personality_mtime = current_mtime
                logger.info("personality_reloaded", hash=self.personality_hash)

        except Exception as e:
            logger.warning("personality_reload_failed", error=str(e))

    def _hash_personality(self, text: str) -> str:
        """Generate hash for personality content."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]

    def _classify_intent(self, message: str) -> str:
        """Classify user intent for model selection."""
        message_lower = message.lower()

        creative_keywords = {'write', 'create', 'story', 'poem', 'creative', 'imagine', 'art'}
        technical_keywords = {'code', 'program', 'debug', 'fix', 'technical', 'algorithm'}

        if any(keyword in message_lower for keyword in creative_keywords):
            return 'creative'
        elif any(keyword in message_lower for keyword in technical_keywords):
            return 'technical'

        return 'concise'

    def _choose_context_model(self, message: str) -> tuple[str, str]:
        """Choose optimal provider and model based on context."""
        if not self.auto_model_enabled:
            provider, model = self._parse_model_string(self.model)
            return provider, model

        intent = self._classify_intent(message)

        preferences = {
            'creative': [
                ('anthropic', 'claude-3-5-sonnet-20241022'),
                ('openai', 'gpt-4o'),
                ('ollama', 'openhermes'),
                ('ollama', 'mistral')
            ],
            'technical': [
                ('openai', 'gpt-4o'),
                ('anthropic', 'claude-3-haiku-20240307'),
                ('ollama', 'mistral'),
                ('ollama', 'openhermes')
            ],
            'concise': [
                ('openai', 'gpt-4o-mini'),
                ('anthropic', 'claude-3-haiku-20240307'),
                ('ollama', 'mistral')
            ]
        }

        for provider_name, model_pattern in preferences.get(intent, []):
            if provider_name in self.available_providers:
                provider = self.providers[provider_name]
                for model in provider.get_models():
                    if model_pattern.lower() in model.lower():
                        full_model_name = f"{provider_name}:{model}"
                        if full_model_name not in self.failed_models:
                            return provider_name, model

        return self._parse_model_string(self.model)

    def _parse_model_string(self, model_string: str) -> tuple[str, str]:
        if ':' in model_string:
            provider, model = model_string.split(':', 1)
            return provider, model
        else:
            return self.current_provider, model_string

    async def generate_response(self, message: str, history: Optional[List[ChatMessage]] = None) -> Dict[str, Any]:
        start_time = time.time()

        try:
            self._maybe_reload_personality()
            provider_name, model_name = self._choose_context_model(message)

            messages = [{"role": "system", "content": self.personality_prompt}]

            if history:
                for msg in history[-10:]:
                    messages.append({"role": msg.role, "content": msg.content})

            messages.append({"role": "user", "content": message})

            provider = self.providers[provider_name]
            response = await provider.chat(
                messages=messages,
                model=model_name,
                temperature=0.7,
                max_tokens=2048
            )

            processing_time = time.time() - start_time
            self.request_count += 1
            self.total_processing_time += processing_time

            full_model_name = f"{provider_name}:{model_name}"
            self.failed_models.discard(full_model_name)

            logger.info(
                "response_generated",
                provider=provider_name,
                model=model_name,
                processing_time=processing_time,
                message_length=len(message),
                response_length=len(response['content'])
            )

            return {
                "response": response['content'],
                "model": full_model_name,
                "model_used": full_model_name,
                "provider": provider_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "processing_time": processing_time
            }

        except Exception as e:
            processing_time = time.time() - start_time
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in ('resource', 'memory', 'timeout', 'overload')):
                full_model_name = f"{provider_name}:{model_name}"
                self.failed_models.add(full_model_name)
                logger.warning("model_marked_failed", model=full_model_name, error=str(e))

            logger.error(
                "response_generation_failed",
                provider=provider_name if 'provider_name' in locals() else 'unknown',
                model=model_name if 'model_name' in locals() else 'unknown',
                error=str(e),
                processing_time=processing_time
            )

            raise HTTPException(
                status_code=500,
                detail={
                    "status": "error",
                    "message": "Failed to generate response",
                    "error": str(e),
                    "provider": provider_name if 'provider_name' in locals() else 'unknown',
                    "model": model_name if 'model_name' in locals() else 'unknown',
                    "processing_time": processing_time
                }
            )

    def metrics(self) -> Dict[str, Any]:
        avg_processing_time = (
            self.total_processing_time / self.request_count
            if self.request_count > 0 else 0.0
        )

        return {
            "active_model": self.model,
            "preferred_model": self.preferred_model,
            "available_models": self.available_models,
            "failed_models": sorted(self.failed_models),
            "auto_model_enabled": self.auto_model_enabled,
            "personality_hash": self.personality_hash,
            "request_count": self.request_count,
            "avg_processing_time": round(avg_processing_time, 3),
            "cache_ttl": self.model_cache_ttl,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    def toggle_auto_model(self, enabled: Optional[bool] = None) -> bool:
        if enabled is not None:
            self.auto_model_enabled = enabled
        else:
            self.auto_model_enabled = not self.auto_model_enabled
        return self.auto_model_enabled


spectra = SpectraAI()

app = FastAPI(
    title="Spectra AI API",
    description="Emotionally intelligent AI assistant backend",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

allowed_origins = [o.strip() for o in os.getenv('ALLOWED_ORIGINS','http://localhost:3000').split(',') if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=allowed_origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"],)

@app.get('/', response_model=Dict[str, Any])
async def root():
    return {
        "service": "Spectra AI Backend API",
        "status": "running",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "frontend_url": f"http://localhost:3000",
        "model": spectra.model,
        "available_models": spectra.available_models,
        "docs": "/docs",
        "health": "/health"
    }

@app.get('/health')
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat(), "personality_hash": spectra.personality_hash}

@app.get('/api/status', response_model=StatusResponse)
async def get_status():
    try:
        current_models = spectra.available_models
        ai_status = "connected" if spectra.available_providers else "disconnected"

        return StatusResponse(
            status="healthy",
            ai_provider=f"multi-provider ({','.join(spectra.available_providers)})",
            ollama_status=ai_status,
            model=spectra.model,
            available_models=current_models,
            timestamp=datetime.now(timezone.utc).isoformat(),
            host=os.getenv('HOST', '127.0.0.1'),
            port=int(os.getenv('PORT', 5000))
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/api/models', response_model=ModelListResponse)
async def list_models():
    spectra.refresh_models()
    return ModelListResponse(
        current=spectra.model,
        available=spectra.available_models,
        preferred=spectra.preferred_model,
        timestamp=datetime.now(timezone.utc).isoformat()
    )

@app.post('/api/models/select', response_model=ModelSelectResponse)
async def select_model(payload: ModelSelectRequest):
    prev = spectra.model
    selected = spectra.set_model(payload.model)
    msg = 'model updated' if selected != prev else 'model unchanged'
    return ModelSelectResponse(
        status='ok',
        selected=selected,
        previous=prev,
        available=spectra.available_models,
        message=msg,
        timestamp=datetime.now(timezone.utc).isoformat()
    )

@app.post('/api/models/refresh', response_model=ModelListResponse)
async def refresh_models_endpoint():
    spectra.refresh_models()
    return ModelListResponse(
        current=spectra.model,
        available=spectra.available_models,
        preferred=spectra.preferred_model,
        timestamp=datetime.now(timezone.utc).isoformat()
    )

@app.post('/api/chat', response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatRequest):
    try:
        logger.info(
            "chat_request",
            preview=chat_request.message[:50],
            history=len(chat_request.history or []),
        )
        result = await spectra.generate_response(chat_request.message, chat_request.history)
        return ChatResponse.build(
            response=result["response"],
            model=result["model"],
            processing_time=result["processing_time"],
        )
    except Exception as e:
        logger.error("chat_error", error=str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "response": "I'm having trouble processing your message right now. Please try again. 💜",
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

@app.get('/api/metrics', response_model=Dict[str, Any])
async def metrics_endpoint():
    return spectra.metrics()

@app.post('/api/auto-model', response_model=Dict[str, Any])
async def toggle_auto_model(req: ToggleAutoModelRequest):
    new_value = spectra.toggle_auto_model(req.enabled)
    return {"auto_model_enabled": new_value, "timestamp": datetime.now(timezone.utc).isoformat()}

@app.get('/api/personality/hash', response_model=Dict[str,str])
async def personality_hash():
    return {"personality_hash": spectra.personality_hash}

@app.post('/api/personality/reload', response_model=Dict[str,str])
async def personality_reload():
    before = spectra.personality_hash
    spectra._maybe_reload_personality()  # Internal call to force reload
    changed = before != spectra.personality_hash
    return {"personality_hash": spectra.personality_hash, "changed": str(changed).lower()}

@app.get('/api/debug/state', response_model=Dict[str, Any])
async def debug_state():
    spectra.refresh_models()
    spectra._maybe_reload_personality()
    base = spectra.metrics()
    base.update({
        "auto_model_enabled": spectra.auto_model_enabled,
        "failed_models_count": len(spectra.failed_models),
        "preferred_model": spectra.preferred_model,
    })
    return base

@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "timestamp": datetime.now(timezone.utc).isoformat()}
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "timestamp": datetime.now(timezone.utc).isoformat()}
    )

if __name__ == '__main__':
    HOST = os.getenv('HOST', '127.0.0.1')
    PORT = int(os.getenv('PORT', 5000))

    logger.info("startup", host=HOST, port=PORT, model=spectra.model, available_models=spectra.available_models, log_format=os.getenv('SPECTRA_LOG_FORMAT', 'json'))

    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=os.getenv('ENVIRONMENT') == 'development',
        log_level="info"
    )

# For Vercel or similar serverless deployment
handler = app