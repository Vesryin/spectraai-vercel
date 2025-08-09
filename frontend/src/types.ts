export interface Message {
  id: string;
  content: string;
  sender: "user" | "spectra";
  timestamp: Date;
  isError?: boolean;
}

export interface ChatResponse {
  response: string;
  error?: string;
}

export interface ApiStatus {
  status: string;
  ai_provider: string;
  personality_loaded: boolean;
}

export interface ModelListResponse {
  current: string;
  available: string[];
  preferred: string;
  timestamp: string;
}

export interface ModelSelectResponse {
  status: string;
  selected: string;
  previous: string;
  available: string[];
  message: string;
  timestamp: string;
}

export interface MetricsResponse {
  active_model: string;
  preferred_model: string;
  failed_models: string[];
  auto_model_enabled: boolean;
  available_models: string[];
  personality_hash: string;
  timestamp: string;
}

export interface ToggleAutoModelResponse {
  auto_model_enabled: boolean;
  timestamp: string;
}

export interface PersonalityHashResponse {
  personality_hash: string;
}
