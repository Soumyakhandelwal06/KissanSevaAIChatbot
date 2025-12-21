# main.py
# FastAPI backend for Farmer Advisory System
# Complete updated version with all fixes and improvements
# ==================== IMPORTS ====================
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator # Corrected to include field_validator
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
import motor.motor_asyncio
from datetime import datetime
import uuid
import logging
from pathlib import Path
import os
import time
from io import BytesIO
import numpy as np
from PIL import Image
import joblib
import pickle
import tensorflow as tf
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel, MarianMTModel, MarianTokenizer
import torch
from collections import defaultdict
from langdetect import detect
from dotenv import load_dotenv
from asyncio import Lock
import hashlib
import math # Added for math functions used in new utilities

# ==================== ENVIRONMENT VARIABLES ====================
load_dotenv()
# Database
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "farmer_advisory_system")
# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
# Rate Limiting
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", 100))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", 60))
# Model Configuration
BASE_DIR = Path(__file__).resolve().parent
MODEL_BASE_PATH_STR = os.getenv("MODEL_BASE_PATH", "./models")
BASE_MODEL_PATH = BASE_DIR / MODEL_BASE_PATH_STR if MODEL_BASE_PATH_STR.startswith(".") else Path(MODEL_BASE_PATH_STR)
# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
# File Upload
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE_MB", "10")) * 1024 * 1024
ALLOWED_EXTENSIONS_STR = os.getenv("ALLOWED_IMAGE_EXTENSIONS", ".jpg,.jpeg,.png,.webp")
ALLOWED_EXTENSIONS = set(ext.strip() for ext in ALLOWED_EXTENSIONS_STR.split(","))
# Model Configuration
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.45"))
ENABLE_GPU = os.getenv("ENABLE_GPU", "true").lower() == "true"
# CORS
ALLOWED_ORIGINS_STR = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173")
ALLOWED_ORIGINS = [origin.strip() for origin in ALLOWED_ORIGINS_STR.split(",")]
# Escalation
ESCALATION_WEBHOOK_URL = os.getenv("ESCALATION_WEBHOOK_URL", "")
ENABLE_ESCALATION_WEBHOOK = os.getenv("ENABLE_ESCALATION_WEBHOOK", "false").lower() == "true"
# Feature Flags
ENABLE_MULTILINGUAL = os.getenv("ENABLE_MULTILINGUAL", "true").lower() == "true"
ENABLE_BATCH_PROCESSING = os.getenv("ENABLE_BATCH_PROCESSING", "true").lower() == "true"
ENABLE_STATISTICS = os.getenv("ENABLE_STATISTICS", "true").lower() == "true"
# ==================== LOGGING ====================
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
# ==================== MULTILINGUAL TRANSLATOR ====================
class MultilingualTranslator:
    def __init__(self):
        self.model_name = {
            'hi-en': 'Helsinki-NLP/opus-mt-hi-en',
            'ml-en': 'Helsinki-NLP/opus-mt-ml-en',
            'en-hi': 'Helsinki-NLP/opus-mt-en-hi',
            'en-ml': 'Helsinki-NLP/opus-mt-en-ml'
        }
        self.tokenizers = {}
        self.models = {}
        self.lang_map = {'hi': 'hi', 'ml': 'ml', 'en': 'en'}
        self.loaded = False
    def load_translation_models(self):
        try:
            logger.info("Loading Multilingual Translation models...")
            for k, v in self.model_name.items():
                self.tokenizers[k] = MarianTokenizer.from_pretrained(v)
                self.models[k] = MarianMTModel.from_pretrained(v)
            self.loaded = True
            logger.info("Multilingual Translation models loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading MarianMT models: {e}. Multilingual support disabled.")
            self.loaded = False
    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        if not self.loaded or src_lang == tgt_lang:
            return text
        key = f"{src_lang}-{tgt_lang}"
        if key not in self.models:
            return text
        try:
            tokenizer = self.tokenizers[key]
            model = self.models[key]
            input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=512)
            # Move to same device as model
            device = next(model.parameters()).device if next(model.parameters(), None) is not None else torch.device("cpu")
            input_ids = input_ids.to(device)
            with torch.no_grad():
                translated = model.generate(input_ids, max_length=512)
            return tokenizer.decode(translated[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return text
    def process_user_input(self, user_input: str, model_response_func):
        # NOTE: This function is still present but the new chat_endpoint
        # now handles translation and intent routing itself for a unified flow.
        if not self.loaded:
            result = model_response_func(user_input)
            return (*result, 'en')
        
        try:
            lang_detected = detect(user_input)
            src_lang = self.lang_map.get(lang_detected, 'en')
        except Exception:
            src_lang = 'en'
        
        if src_lang != 'en':
            user_input_en = self.translate(user_input, src_lang, 'en')
        else:
            user_input_en = user_input
        
        answer_en, confidence, used_model = model_response_func(user_input_en)
        
        if src_lang != 'en':
            answer_local = self.translate(answer_en, 'en', src_lang)
        else:
            answer_local = answer_en
        
        return answer_local, confidence, used_model, src_lang
# ==================== MODEL CONFIG ====================
MODEL_MAP = {
    "insect_classifier": {
        "path": BASE_MODEL_PATH / "insect_image_model_outputs" / "insect_classifier_model.keras",
        "type": "keras",
        "task": "insect_classification",
        "labels_path": BASE_MODEL_PATH / "insect_image_model_outputs" / "class_indices.sav",
        "labels": []
    },
    "disease_classifier": {
        "path": BASE_MODEL_PATH / "disease_classifier.keras",
        "type": "keras",
        "task": "disease_classification",
        "labels": ["bacterial_blight", "blast", "brown_spot", "tungro", "healthy"]
    },
    
    "crop_classifier": {
        "path": BASE_MODEL_PATH / "crop_images_model_outputs" / "crop_classifier_model.keras",
        "type": "keras",
        "task": "crop_classification",
        "labels": ["rice", "wheat", "maize", "cotton", "sugarcane", "millet", "pulses"]
    },
    "farmer_advisory": {
        "path": BASE_MODEL_PATH / "farmer_call_query",
        "type": "huggingface_bert",
        "task": "advisory",
    },
    "faq_retrieval": {
        "path": BASE_MODEL_PATH / "farmer_faq",
        "type": "huggingface_seq2seq",
        "task": "faq"
    },
    "pesticide_recommendation": {
        "path": BASE_MODEL_PATH / "pesticide_solution2.sav",
        "type": "sklearn",
        "task": "pesticide_recommendation",
        "labels": []
    },
    "crop_calendar": {
        "path": BASE_MODEL_PATH / "random_forest_crop_calendar_prediction.sav",
        "type": "sklearn",
        "task": "crop_calendar"
    },
    "fertilizer": {
        "path": BASE_MODEL_PATH / "random_forest_fertilizer_prediction.sav",
        "type": "sklearn",
        "task": "fertilizer"
    },
    "yield_prediction": {
        "path": BASE_MODEL_PATH / "random_forest_yield_pipeline.sav",
        "type": "sklearn",
        "task": "yield"
    },
    "price_clustering": {
        "path": BASE_MODEL_PATH / "kmeans_prices_pipeline.sav",
        "type": "sklearn",
        "task": "price"
    },
    "kmeans_clustering": {
        "path": BASE_MODEL_PATH / "kmeans_clustering_model.sav",
        "type": "sklearn",
        "task": "clustering"
    },
    "rainfall_prediction": {
        "path": BASE_MODEL_PATH / "random_forest_rainfall_prediction.sav",
        "type": "sklearn",
        "task": "rainfall"
    },
    "xgb_model": {
        "path": BASE_MODEL_PATH / "xgb_model.sav",
        "type": "sklearn",
        "task": "general"
    }
}
CONFIDENCE_THRESHOLD = 0.45
# ==================== PYDANTIC MODELS ====================
class ContextData(BaseModel):
    farmer_id: Optional[str] = ""
    location: Optional[str] = ""
    crop: Optional[str] = ""
    season: Optional[str] = ""

# ------------------- Chat endpoint models (NEW) -------------------
class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    context: Optional[Dict[str, Any]] = None
    request_escalation: bool = False

    @field_validator("query")
    def non_empty_query(cls, v):
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()

class ChatResponse(BaseModel):
    query_id: str
    answer: str
    confidence: float
    intent: str
    used_model: str
    detected_language: str
    escalation_id: Optional[str] = None
    timestamp: str
# ------------------------------------------------------------------

class TextQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    context: Optional[ContextData] = None
    request_escalation: bool = False
class FeedbackRequest(BaseModel):
    query_id: str = Field(..., min_length=1)
    rating: int = Field(..., ge=1, le=5)
    feedback_text: Optional[str] = Field(None, max_length=1000)
class CorrectionRequest(BaseModel):
    query_id: str = Field(..., min_length=1)
    correct_label: str = Field(..., min_length=1, max_length=100)
    notes: Optional[str] = Field(None, max_length=500)
class QueryResponse(BaseModel):
    query_id: str
    answer: str
    confidence: float
    used_model: str
    timestamp: str
    escalation_id: Optional[str] = None
class ImageResponse(BaseModel):
    query_id: str
    label: str
    confidence: float
    used_model: str
    timestamp: str
    escalation_id: Optional[str] = None
class PredictionRequest(BaseModel):
    features: Dict[str, Any] = Field(..., description="Feature dictionary for prediction")
class PredictionResponse(BaseModel):
    prediction: Any
    confidence: Optional[float] = None
    model_used: str
    timestamp: str
    model_config = {
        "protected_namespaces": ()
    }
# ==================== MODEL LOADER ====================
class ModelLoader:
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
    def load_keras_model(self, model_path: Path):
        try:
            model = tf.keras.models.load_model(str(model_path), compile=False)
            logger.info(f"Loaded Keras model: {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading Keras model {model_path}: {e}")
            return None
    def load_sklearn_model(self, model_path: Path):
        try:
            # Handle potential use of pickle
            if model_path.suffix == '.pkl':
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            else:
                model = joblib.load(str(model_path))
            logger.info(f"Loaded sklearn model: {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading sklearn model {model_path}: {e}")
            return None
    def load_huggingface_model(self, model_path: Path, model_type: str = "seq2seq"):
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            if model_type == "seq2seq":
                model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path))
            else:
                model = AutoModel.from_pretrained(str(model_path))
            
            # Move to GPU if available
            if ENABLE_GPU:
                if torch.cuda.is_available():
                    model = model.to("cuda")
                elif torch.backends.mps.is_available():
                    model = model.to("mps")
            
            logger.info(f"Loaded HuggingFace {model_type} model: {model_path}")
            return {"model": model, "tokenizer": tokenizer, "type": model_type}
        except Exception as e:
            logger.error(f"Error loading HuggingFace model {model_path}: {e}")
            return None
    def load_all_models(self):
        for model_name, config in MODEL_MAP.items():
            try:
                model_type = config["type"]
                model_path = config["path"]
                
                if not model_path.exists():
                    logger.warning(f"Model path not found: {model_path}")
                    continue
                
                if model_type == "keras":
                    self.models[model_name] = self.load_keras_model(model_path)
                elif model_type == "sklearn":
                    self.models[model_name] = self.load_sklearn_model(model_path)
                elif model_type == "huggingface_seq2seq":
                    self.models[model_name] = self.load_huggingface_model(model_path, "seq2seq")
                elif model_type == "huggingface_bert":
                    self.models[model_name] = self.load_huggingface_model(model_path, "bert")
                
                # Load labels for insect classifier
                if model_name == "insect_classifier":
                    labels_path = config.get("labels_path")
                    if labels_path and labels_path.exists():
                        with open(labels_path, "rb") as f:
                            idx_map = pickle.load(f)
                            config["labels"] = list(idx_map.keys()) if isinstance(idx_map, dict) else list(idx_map)
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
        
        logger.info(f"Loaded {len(self.models)} models successfully")
    def get_model(self, model_name: str):
        return self.models.get(model_name)
# ==================== RATE LIMITER ====================
class RateLimiter:
    def __init__(self, requests: int, window: int):
        self.requests = requests
        self.window = window
        self.clients = defaultdict(list)
        self.lock = Lock()
    async def is_allowed(self, client_id: str) -> bool:
        async with self.lock:
            now = time.time()
            self.clients[client_id] = [t for t in self.clients[client_id] if now - t < self.window]
            
            if len(self.clients[client_id]) >= self.requests:
                return False
            
            self.clients[client_id].append(now)
            return True
# ==================== GLOBAL INSTANCES ====================
model_loader = ModelLoader()
rate_limiter = RateLimiter(RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW)
translator = MultilingualTranslator()
db_client = None
db = None
# ==================== LIFESPAN ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_client, db
    
    # Connect to MongoDB
    try:
        db_client = motor.motor_asyncio.AsyncIOMotorClient(
            MONGODB_URL,
            serverSelectionTimeoutMS=5000
        )
        # Test connection
        await db_client.admin.command('ping')
        db = db_client[DB_NAME]
        logger.info("Connected to MongoDB successfully")
    except Exception as e:
        logger.error(f"MongoDB connection failed: {e}. Running without database.")
        db = None
    
    # Load models
    model_loader.load_all_models()
    translator.load_translation_models()
    
    # Pre-cache intent prototypes after models are loaded
    bert_bundle = model_loader.get_model("farmer_advisory")
    if bert_bundle:
        prepare_intent_prototypes(bert_bundle)
        logger.info("Intent prototypes pre-cached.")
    
    yield
    
    # Cleanup
    if db_client:
        db_client.close()
        logger.info("Closed MongoDB connection")
# ==================== FASTAPI APP ====================
app = FastAPI(
    title="Farmer Advisory System API",
    description="AI-powered agricultural advisory system with multilingual support",
    version="2.0.0",
    lifespan=lifespan
)
# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)
# Rate Limiting Middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host if request.client else "unknown"
    
    # Skip rate limiting for health check
    if request.url.path == "/health":
        return await call_next(request)
    
    if not await rate_limiter.is_allowed(client_ip):
        return JSONResponse(
            status_code=429,
            content={"detail": "Too many requests. Please try again later."}
        )
    
    return await call_next(request)
# ==================== HELPER FUNCTIONS ====================

# ------------------- Embedding / Intent utilities (NEW) -------------------
# This caches intent prototype embeddings on first use.
_intent_proto_cache = {"embeddings": None, "labels": None}

def _mean_pooling(last_hidden_state, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def encode_text_with_bert(text: str, bert_model_bundle):
    """
    Encodes text using the loaded farmer_call_query model (AutoModel + tokenizer).
    Returns a normalized numpy embedding.
    """
    if bert_model_bundle is None:
        return None

    tokenizer = bert_model_bundle["tokenizer"]
    model = bert_model_bundle["model"]

    # Use model's device or CPU fallback
    device = next(model.parameters()).device if next(model.parameters(), None) is not None else torch.device("cpu")
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        # last_hidden_state present for AutoModel; do mean pooling
        last_hidden = outputs.last_hidden_state # (1, seq_len, hidden)
        pooled = _mean_pooling(last_hidden, inputs["attention_mask"]) # (1, hidden)
        emb = pooled[0].cpu().numpy()
        # Normalize the embedding vector
        norm = emb / (np.linalg.norm(emb) + 1e-12)
        return norm

def prepare_intent_prototypes(bert_bundle):
    """
    Define intent prototypes (short example phrases per intent),
    encode them and cache embeddings. This runs on-first-use.
    """
    if _intent_proto_cache["embeddings"] is not None:
        return

    # Define human-readable intent names and example phrases
    intents = {
        "faq": ["what is the best fertilizer for rice", "how to plant maize", "what are common diseases in wheat"],
        "general_advice": ["advise on farming", "recommend farming best practices", "general agricultural advice"],
        "fertilizer": ["which fertilizer to use", "recommend fertilizer for my crop", "NPK suggestions"],
        "yield": ["predict my yield", "expected yield for this season"],
        "rainfall": ["is it going to rain", "rain forecast for next week"],
        "pesticide": ["which pesticide to use", "pesticide recommendation for pests"],
        "crop_recommendation": ["what crop should i grow", "best crop for this soil and season"],
        "price": ["market price for rice", "current mandi rates"],
        "image_disease": ["my leaves have spots", "photo of diseased leaf", "plant leaf disease image"],
        "image_insect": ["i see insects on leaves", "bugs in my crop picture", "photo of pest"],
        "image_crop": ["identify this crop", "what crop is this in the photo"],
        "unknown": ["i don't know", "not sure"]
    }

    labels = list(intents.keys())
    # combine first two examples to stabilize prototype embedding
    phrases = [" ".join(intents[label][:2]) for label in labels]
    embeddings = []
    for p in phrases:
        emb = encode_text_with_bert(p, bert_bundle)
        # Assuming BERT dim is 768, replace with actual dim if necessary
        if emb is None:
            embeddings.append(np.zeros(768, dtype=float))
        else:
            embeddings.append(emb)
    
    if not embeddings or embeddings[0].ndim == 0:
        logger.error("Failed to generate intent embeddings. Intent classification will be non-functional.")
        return

    _intent_proto_cache["embeddings"] = np.vstack(embeddings) # (num_intents, dim)
    _intent_proto_cache["labels"] = labels

def classify_intent(text_en: str, bert_bundle):
    """
    Returns (intent_label, confidence_score) using cosine similarity against prototypes.
    """
    try:
        # Prototypes are pre-cached in lifespan, but this ensures fallback if necessary
        prepare_intent_prototypes(bert_bundle)
        
        emb = encode_text_with_bert(text_en, bert_bundle)
        if emb is None or _intent_proto_cache["embeddings"] is None:
            return "unknown", 0.0
            
        proto = _intent_proto_cache["embeddings"]
        # cosine because both vectors (proto and emb) are normalized
        sims = np.dot(proto, emb)
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        label = _intent_proto_cache["labels"][best_idx]
        
        # Normalize similarity from [-1,1] to [0,1] for confidence
        confidence = (best_score + 1.0) / 2.0
        return label, confidence
    except Exception as e:
        logger.error(f"Intent classification error: {e}", exc_info=True)
        return "unknown", 0.0
# -------------------------------------------------------------------------

def preprocess_image(image: Image.Image, target_size=(224, 224)) -> np.ndarray:
    """Preprocess image for model prediction with validation"""
    # Validate image dimensions
    if image.size[0] < 50 or image.size[1] < 50:
        raise ValueError("Image too small. Minimum size is 50x50 pixels.")
    if image.size[0] > 4000 or image.size[1] > 4000:
        raise ValueError("Image too large. Maximum size is 4000x4000 pixels.")
    
    # Resize and normalize
    image = image.convert('RGB').resize(target_size, Image.Resampling.LANCZOS)
    img_array = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)
async def create_escalation(query_id: str, query: str, confidence: float, context: Optional[ContextData]):
    """Create escalation record for low confidence predictions"""
    escalation_id = str(uuid.uuid4())
    
    # Handle Dict[str, Any] context from ChatRequest
    if isinstance(context, dict):
        context_data = context
    elif isinstance(context, ContextData):
        context_data = context.dict()
    else:
        context_data = {}

    if db:
        try:
            escalation_doc = {
                "escalation_id": escalation_id,
                "query_id": query_id,
                "query": query,
                "confidence": confidence,
                "context": context_data,
                "status": "pending",
                "created_at": datetime.utcnow().isoformat()
            }
            await db.escalations.insert_one(escalation_doc)
            
            # Send webhook notification if enabled
            if ENABLE_ESCALATION_WEBHOOK and ESCALATION_WEBHOOK_URL:
                try:
                    import aiohttp
                    async with aiohttp.ClientSession() as session:
                        await session.post(
                            ESCALATION_WEBHOOK_URL,
                            json=escalation_doc,
                            timeout=aiohttp.ClientTimeout(total=5)
                        )
                except Exception as e:
                    logger.warning(f"Webhook notification failed: {e}")
        except Exception as e:
            logger.error(f"Error creating escalation: {e}")
    
    return escalation_id
def predict_image(model_name: str, image: Image.Image):
    """Run image classification with specified model"""
    model = model_loader.get_model(model_name)
    if model is None:
        return None, 0.0
    
    config = MODEL_MAP.get(model_name)
    if config is None:
        return None, 0.0
    
    try:
        img_array = preprocess_image(image)
        preds = model.predict(img_array, verbose=0)
        
        if preds.ndim > 1:
            preds = preds[0]
        
        label_idx = int(np.argmax(preds))
        confidence = float(np.max(preds))
        
        labels = config.get("labels", [])
        label = labels[label_idx] if labels and label_idx < len(labels) else str(label_idx)
        
        return label, confidence
    except Exception as e:
        logger.error(f"Image prediction error with {model_name}: {e}")
        return None, 0.0
def get_model_response(query_en: str) -> tuple:
    """
    Get response from appropriate model based on query (Used for FAQ/Advisory fallback).
    This function is used in the new chat_endpoint for routing purposes.
    """
    # Try FAQ retrieval model first
    faq_model = model_loader.get_model("faq_retrieval")
    if faq_model:
        try:
            tokenizer = faq_model["tokenizer"]
            model = faq_model["model"]
            
            # Prepare input
            inputs = tokenizer(
                query_en,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            # Move to same device as model
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=150,
                    num_beams=4,
                    early_stopping=True
                )
            
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Calculate confidence (simplified - based on output length and coherence)
            confidence = min(0.85, 0.5 + len(answer.split()) / 100)
            
            return answer, confidence, "faq_retrieval"
        except Exception as e:
            logger.error(f"FAQ model error: {e}")
    
    # Try advisory model as fallback
    advisory_model = model_loader.get_model("farmer_advisory")
    if advisory_model:
        try:
            # Placeholder advisory logic (replace with your actual BER based Q&A or advisory generation)
            answer = "Based on your general query, I recommend checking local advisories and soil health reports for best practices."
            confidence = 0.65
            
            return answer, confidence, "farmer_advisory_placeholder"
        except Exception as e:
            logger.error(f"Advisory model error: {e}")
    
    # Fallback response
    return (
        "I understand your query but need more information to provide specific advice. Please provide details about your crop, location, and specific concerns.",
        0.40,
        "fallback"
    )
# ==================== ENDPOINTS ====================
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": len(model_loader.models),
        "translator_loaded": translator.loaded,
        "database_connected": db is not None,
        "timestamp": datetime.utcnow().isoformat()
    }

# ------------------- Chat endpoint (NEW) -------------------
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Unified chat endpoint:
      1) detect language
      2) translate -> en
      3) classify intent (farmer_call_query BERT)
      4) route to appropriate model/handler
      5) translate back to user language
      6) return ChatResponse
    """
    query_id = str(uuid.uuid4())
    user_text = request.query
    
    # 1) Detect language
    if ENABLE_MULTILINGUAL:
        try:
            detected_lang = detect(user_text)
        except Exception:
            detected_lang = "en"
    else:
        detected_lang = "en"
        
    src_lang = detected_lang if detected_lang in translator.lang_map else "en"

    # 2) Translate user -> English (if needed)
    if translator.loaded and src_lang != "en":
        try:
            user_text_en = translator.translate(user_text, src_lang, "en")
        except Exception as e:
            logger.error(f"Translation to English failed: {e}")
            user_text_en = user_text
    else:
        user_text_en = user_text

    # 3) Intent classification using farmer_call_query
    bert_bundle = model_loader.get_model("farmer_advisory")  # returns {"model","tokenizer","type"} or None
    intent_label, intent_conf = classify_intent(user_text_en, bert_bundle)

    used_model = "intent_classifier:farmer_advisory"
    answer_en = ""
    answer_conf = 0.0
    escalation_id = None

    # 4) Route by intent
    # 4a. Image related intents: instruct user to upload image
    if intent_label in ("image_disease", "image_insect", "image_crop"):
        answer_en = (
            "It looks like you want an image-based diagnosis. "
            "Please upload a clear photo of the plant using the Image tab so I can analyze it."
        )
        answer_conf = intent_conf
        used_model = "instruction:image_upload"
    else:
        # 4b. Textual / FAQ / Advisory intents → use seq2seq FAQ/advisory pipeline
        if intent_label in ("faq", "general_advice", "unknown", "price"):
            # Leverage your existing get_model_response which tries faq_retrieval then advisory fallback
            try:
                answer_en, answer_conf, model_used = get_model_response(user_text_en)
                used_model = model_used
            except Exception as e:
                logger.error(f"Error getting model response: {e}")
                answer_en = "Sorry, I couldn't generate an answer right now."
                answer_conf = 0.0
                used_model = "fallback"

        # 4c. Structured prediction intents
        elif intent_label in ("fertilizer", "yield", "rainfall", "pesticide", "crop_recommendation"):
            # These require structured inputs. Try to extract 'features' from context
            ctx = request.context or {}
            # Allow 'features' key in context dictionary, or directly use the whole context if it looks like features
            features = ctx.get("features") or ctx.get("feature_vector") or None
            
            # If the context is passed as a flat dictionary of features
            if not features and all(isinstance(v, (int, float)) for v in ctx.values()):
                features = list(ctx.values())
            
            if features and isinstance(features, (list, tuple)):
                # map intent -> model key in MODEL_MAP
                intent_to_model = {
                    "fertilizer": "fertilizer",
                    "yield": "yield_prediction",
                    "rainfall": "rainfall_prediction",
                    "pesticide": "pesticide_recommendation",
                    "crop_recommendation": "kmeans_clustering"  # fallback
                }
                model_key = intent_to_model.get(intent_label)
                model = model_loader.get_model(model_key)
                
                if model is None:
                    answer_en = f"Sorry, the required prediction model for {intent_label} is currently unavailable."
                    answer_conf = 0.0
                    used_model = model_key or "unknown"
                else:
                    try:
                        # sklearn style: features must be a list of lists or array
                        pred = model.predict([list(features)])[0]
                        answer_en = f"Prediction for {intent_label}: {pred}"
                        answer_conf = 0.8
                        used_model = model_key
                    except Exception as e:
                        logger.error(f"Structured prediction error for {model_key}: {e}", exc_info=True)
                        answer_en = "Failed to compute prediction. Please ensure correct numeric features are provided in the context."
                        answer_conf = 0.0
                        used_model = model_key
            else:
                # Missing structured inputs — ask a follow-up
                answer_en = (
                    "To provide a prediction, I need some numeric features (e.g., soil NPK, pH, area, season) "
                    f"relevant to {intent_label}. Please include these details in your context or query."
                )
                answer_conf = 0.0
                used_model = "needs_context"

        else:
            # Final Fallback (Should be caught by 'unknown' intent, but this is a safety net)
            try:
                answer_en, answer_conf, model_used = get_model_response(user_text_en)
                used_model = model_used
            except Exception as e:
                logger.error(f"Final fallback model error: {e}")
                answer_en = "Sorry, I couldn't find a suitable answer."
                answer_conf = 0.0
                used_model = "fallback"

    # 5) Escalation if needed
    if answer_conf < CONFIDENCE_THRESHOLD or request.request_escalation:
        try:
            # Pass ContextData fields or the raw Dict[str, Any] context
            escalation_id = await create_escalation(query_id, user_text, float(answer_conf), request.context)
        except Exception as e:
            logger.error(f"Escalation creation failed: {e}")
            escalation_id = None

    # 6) Translate answer back to user language if needed
    if translator.loaded and src_lang != "en":
        try:
            answer_local = translator.translate(answer_en, "en", src_lang)
        except Exception as e:
            logger.error(f"Translation back to local failed: {e}")
            answer_local = answer_en
    else:
        answer_local = answer_en

    # 7) Save query to DB
    doc = {
        "query_id": query_id,
        "query": user_text,
        "query_en": user_text_en,
        "answer": answer_local,
        "answer_en": answer_en,
        "confidence": float(answer_conf),
        "used_model": used_model,
        "intent": intent_label,
        "detected_language": src_lang,
        "escalation_id": escalation_id,
        "context": request.context or {},
        "timestamp": datetime.utcnow().isoformat()
    }
    if db:
        try:
            # Use 'chat_queries' collection for this new endpoint's data
            await db.chat_queries.insert_one(doc)
        except Exception as e:
            logger.error(f"DB insert failure for chat: {e}")

    # 8) Build response
    resp = ChatResponse(
        query_id=query_id,
        answer=answer_local,
        confidence=float(answer_conf),
        intent=intent_label,
        used_model=used_model,
        detected_language=src_lang,
        escalation_id=escalation_id,
        timestamp=doc["timestamp"]
    )
    return resp
# -------------------------------------------------------------------

# NOTE: The original /api/query endpoint is removed as /api/chat is its replacement.
# @app.post("/api/query", response_model=QueryResponse)
# async def process_text_query(request: TextQueryRequest):
#     ... [REMOVED IN FAVOR OF /api/chat]

@app.post("/api/image", response_model=ImageResponse)
async def classify_image(
    file: UploadFile = File(...),
    context: Optional[str] = None
):
    """Classify agricultural images (insects, diseases, crops)"""
    query_id = str(uuid.uuid4())
    
    try:
        # Validate file extension
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        # Read and validate file size
        image_bytes = await file.read()
        if len(image_bytes) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE / 1024 / 1024}MB"
            )
        
        if len(image_bytes) < 100:
            raise HTTPException(
                status_code=400,
                detail="File too small or corrupted"
            )
        
        # Open and validate image
        try:
            image = Image.open(BytesIO(image_bytes))
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid or corrupted image file")
        
        # Try all image classification models and pick best result
        best_label = None
        best_conf = 0.0
        best_model = None
        
        for model_name in ["insect_classifier", "disease_classifier", "crop_classifier"]:
            label, conf = predict_image(model_name, image)
            if label and conf > best_conf:
                best_conf = conf
                best_label = label
                best_model = model_name
        
        if best_label is None:
            raise HTTPException(
                status_code=500,
                detail="Unable to classify image. Please try a different image."
            )
        
        # Create escalation if confidence is low
        escalation_id = None
        if best_conf < CONFIDENCE_THRESHOLD:
            context_data = ContextData() if not context else ContextData(**eval(context))
            # Use ContextData for image escalation since it's the model defined for this endpoint's context
            escalation_id = await create_escalation(
                query_id,
                f"Image classification: {file.filename}",
                best_conf,
                context_data
            )
        
        # Store in database
        doc = {
            "query_id": query_id,
            "file_name": file.filename,
            "label": best_label,
            "confidence": best_conf,
            "used_model": best_model,
            "escalation_id": escalation_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if db:
            try:
                await db.image_queries.insert_one(doc)
            except Exception as e:
                logger.error(f"Database insert error: {e}")
        
        return ImageResponse(**doc)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image classification error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing the image. Please try again."
        )
@app.post("/api/predict/fertilizer", response_model=PredictionResponse)
async def predict_fertilizer(request: PredictionRequest):
    """Predict fertilizer recommendations"""
    # Validate features
    if not request.features:
        raise HTTPException(status_code=400, detail="Features dictionary cannot be empty")
    
    model = model_loader.get_model("fertilizer")
    if not model:
        raise HTTPException(status_code=503, detail="Fertilizer model not available")
    
    try:
        # Convert features to appropriate format
        feature_values = list(request.features.values())
        prediction = model.predict([feature_values])[0]
        
        return PredictionResponse(
            prediction=str(prediction),
            model_used="fertilizer",
            timestamp=datetime.utcnow().isoformat()
        )
    except Exception as e:
        logger.error(f"Fertilizer prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")
@app.post("/api/predict/yield", response_model=PredictionResponse)
async def predict_yield(request: PredictionRequest):
    """Predict crop yield"""
    model = model_loader.get_model("yield_prediction")
    if not model:
        raise HTTPException(status_code=503, detail="Yield prediction model not available")
    
    try:
        feature_values = list(request.features.values())
        prediction = model.predict([feature_values])[0]
        
        return PredictionResponse(
            prediction=float(prediction),
            model_used="yield_prediction",
            timestamp=datetime.utcnow().isoformat()
        )
    except Exception as e:
        logger.error(f"Yield prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")
@app.post("/api/predict/rainfall", response_model=PredictionResponse)
async def predict_rainfall(request: PredictionRequest):
    """Predict rainfall"""
    model = model_loader.get_model("rainfall_prediction")
    if not model:
        raise HTTPException(status_code=503, detail="Rainfall prediction model not available")
    
    try:
        feature_values = list(request.features.values())
        prediction = model.predict([feature_values])[0]
        
        return PredictionResponse(
            prediction=float(prediction),
            model_used="rainfall_prediction",
            timestamp=datetime.utcnow().isoformat()
        )
    except Exception as e:
        logger.error(f"Rainfall prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")
@app.post("/api/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """Submit feedback for a query"""
    if not db:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        await db.feedback.insert_one({
            "query_id": feedback.query_id,
            "rating": feedback.rating,
            "feedback_text": feedback.feedback_text,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return {"status": "success", "message": "Feedback submitted successfully"}
    except Exception as e:
        logger.error(f"Feedback submission error: {e}")
        raise HTTPException(status_code=500, detail="Failed to submit feedback")
@app.post("/api/correction")
async def submit_correction(correction: CorrectionRequest):
    """Submit correction for a prediction"""
    if not db:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        await db.corrections.insert_one({
            "query_id": correction.query_id,
            "correct_label": correction.correct_label,
            "notes": correction.notes,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return {"status": "success", "message": "Correction submitted successfully"}
    except Exception as e:
        logger.error(f"Correction submission error: {e}")
        raise HTTPException(status_code=500, detail="Failed to submit correction")
@app.get("/api/models")
async def list_models():
    """List all available models and their status"""
    models_status = {}
    
    for model_name, config in MODEL_MAP.items():
        model = model_loader.get_model(model_name)
        models_status[model_name] = {
            "loaded": model is not None,
            "type": config["type"],
            "task": config["task"],
            "path_exists": config["path"].exists()
        }
    
    return {
        "models": models_status,
        "total_loaded": len(model_loader.models),
        "translator_available": translator.loaded
    }
@app.get("/api/queries/{query_id}")
async def get_query(query_id: str):
    """Retrieve a specific query by ID (Looks in chat_queries first, then older queries)"""
    if not db:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        # Check new chat_queries collection
        query = await db.chat_queries.find_one({"query_id": query_id}, {"_id": 0})
        if not query:
            # Check older queries collection (assuming the old /api/query stored there)
            query = await db.queries.find_one({"query_id": query_id}, {"_id": 0})
        
        if not query:
            raise HTTPException(status_code=404, detail="Query not found")
        
        return query
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query retrieval error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve query")
@app.get("/api/queries")
async def list_queries(
    limit: int = 10,
    skip: int = 0,
    farmer_id: Optional[str] = None
):
    """List recent queries with optional filtering (from the new chat_queries collection)"""
    if not db:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        filter_query = {}
        # Assuming the new chat_endpoint uses "context.farmer_id" structure if context is properly populated
        if farmer_id:
            filter_query["context.farmer_id"] = farmer_id
        
        cursor = db.chat_queries.find(filter_query, {"_id": 0}).sort("timestamp", -1).skip(skip).limit(limit)
        queries = await cursor.to_list(length=limit)
        
        return {
            "queries": queries,
            "count": len(queries),
            "skip": skip,
            "limit": limit
        }
    except Exception as e:
        logger.error(f"Query listing error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve queries")
@app.get("/api/escalations")
async def list_escalations(status: Optional[str] = None, limit: int = 20):
    """List escalations with optional status filter"""
    if not db:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        filter_query = {}
        if status:
            filter_query["status"] = status
        
        cursor = db.escalations.find(filter_query, {"_id": 0}).sort("created_at", -1).limit(limit)
        escalations = await cursor.to_list(length=limit)
        
        return {
            "escalations": escalations,
            "count": len(escalations)
        }
    except Exception as e:
        logger.error(f"Escalations listing error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve escalations")
@app.put("/api/escalations/{escalation_id}")
async def update_escalation(escalation_id: str, status: str, resolution: Optional[str] = None):
    """Update escalation status"""
    if not db:
        raise HTTPException(status_code=503, detail="Database not available")
    
    if status not in ["pending", "in_progress", "resolved", "rejected"]:
        raise HTTPException(status_code=400, detail="Invalid status")
    
    try:
        update_doc = {
            "status": status,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        if resolution:
            update_doc["resolution"] = resolution
        
        result = await db.escalations.update_one(
            {"escalation_id": escalation_id},
            {"$set": update_doc}
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Escalation not found")
        
        return {"status": "success", "message": "Escalation updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Escalation update error: {e}")
        raise HTTPException(status_code=500, detail="Failed to update escalation")
@app.get("/api/statistics")
async def get_statistics():
    """Get system statistics"""
    if not ENABLE_STATISTICS:
        raise HTTPException(status_code=403, detail="Statistics endpoint is disabled")
    
    if not db:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        stats = {
            # Use the new collection for total queries
            "total_queries": await db.chat_queries.count_documents({}),
            "total_image_queries": await db.image_queries.count_documents({}),
            "total_escalations": await db.escalations.count_documents({}),
            "pending_escalations": await db.escalations.count_documents({"status": "pending"}),
            "total_feedback": await db.feedback.count_documents({}),
            "total_corrections": await db.corrections.count_documents({}),
        }
        
        # Average confidence
        pipeline = [
            {"$group": {"_id": None, "avg_confidence": {"$avg": "$confidence"}}}
        ]
        # Use the new collection for confidence stats
        result = await db.chat_queries.aggregate(pipeline).to_list(1)
        stats["average_confidence"] = result[0]["avg_confidence"] if result else 0
        
        # Model usage distribution
        model_usage_pipeline = [
            {"$group": {"_id": "$used_model", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        # Use the new collection for model usage stats
        model_usage = await db.chat_queries.aggregate(model_usage_pipeline).to_list(10)
        stats["model_usage"] = {item["_id"]: item["count"] for item in model_usage}
        
        return stats
    except Exception as e:
        logger.error(f"Statistics error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve statistics")
@app.delete("/api/queries/{query_id}")
async def delete_query(query_id: str):
    """Delete a query (admin only - add auth in production)"""
    if not db:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        # Delete from both potential collections for completeness
        result = await db.chat_queries.delete_one({"query_id": query_id})
        if result.deleted_count == 0:
            result = await db.queries.delete_one({"query_id": query_id})
            
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Query not found")
        
        return {"status": "success", "message": "Query deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query deletion error: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete query")
@app.post("/api/batch/images")
async def batch_classify_images(files: List[UploadFile] = File(...)):
    """Batch process multiple images"""
    if not ENABLE_BATCH_PROCESSING:
        raise HTTPException(status_code=403, detail="Batch processing is disabled")
    
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images allowed per batch")
    
    results = []
    
    for file in files:
        try:
            # Validate file
            ext = Path(file.filename).suffix.lower()
            if ext not in ALLOWED_EXTENSIONS:
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "message": "Invalid file type"
                })
                continue
            
            image_bytes = await file.read()
            if len(image_bytes) > MAX_FILE_SIZE:
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "message": "File too large"
                })
                continue
            
            image = Image.open(BytesIO(image_bytes))
            
            # Classify
            best_label = None
            best_conf = 0.0
            best_model = None
            
            for model_name in ["insect_classifier", "disease_classifier", "crop_classifier"]:
                label, conf = predict_image(model_name, image)
                if label and conf > best_conf:
                    best_conf = conf
                    best_label = label
                    best_model = model_name
            
            results.append({
                "filename": file.filename,
                "status": "success",
                "label": best_label,
                "confidence": best_conf,
                "model": best_model
            })
        
        except Exception as e:
            logger.error(f"Batch processing error for {file.filename}: {e}")
            results.append({
                "filename": file.filename,
                "status": "error",
                "message": str(e)
            })
    
    return {"results": results, "total": len(files), "processed": len(results)}
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Farmer Advisory System API",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "chat": "/api/chat", # Updated from /api/query
            "image": "/api/image",
            "models": "/api/models",
            "statistics": "/api/statistics"
        },
        "documentation": "/docs"
    }
# ==================== ERROR HANDLERS ====================
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle validation errors"""
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)}
    )
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again later."}
    )
# ==================== RUN SERVER ====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=API_HOST,
        port=API_PORT,
        log_level=LOG_LEVEL.lower()
    )