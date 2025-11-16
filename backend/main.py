# main.py
# FastAPI backend for Farmer Advisory System
# Updated: Integrates MultilingualTranslator for Hindi and Malayalam support.

# ==================== IMPORTS ====================
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
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
# tensorflow import left as is for keras models (if using mac, use tensorflow-macos)
import tensorflow as tf
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MarianMTModel, MarianTokenizer # ADDED MARIAN IMPORTS
# SentenceTransformer kept in case you ever use a sentence-transformer style retrieval model
from sentence_transformers import SentenceTransformer
import torch
import json
import asyncio
from collections import defaultdict
from langdetect import detect # ADDED LANGDETECT IMPORT


# ==================== ENVIRONMENT VARIABLES ====================
from dotenv import load_dotenv
import os
from pathlib import Path

# Load .env file at the very beginning
load_dotenv()

# MongoDB configuration
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DB_NAME = "farmer_advisory_system"

# Webhook URL
ESCALATION_WEBHOOK_URL = os.getenv("ESCALATION_WEBHOOK_URL", "https://example.com/webhook/escalation")

# API settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))

# Rate limiting
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", 100))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", 60))  # in seconds

# Model base path
BASE_DIR = Path(__file__).resolve().parent
BASE_MODEL_PATH = BASE_DIR / "models"

# Logging level
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


# ==================== LOGGING SETUP ====================
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== MULTILINGUAL TRANSLATOR CLASS ====================

class MultilingualTranslator:
    """Handles translation between Hindi/Malayalam and English."""
    def __init__(self):
        # Define supported models
        self.model_name = {
            'hi-en': 'Helsinki-NLP/opus-mt-hi-en',
            'ml-en': 'Helsinki-NLP/opus-mt-ml-en',
            'en-hi': 'Helsinki-NLP/opus-mt-en-hi',
            'en-ml': 'Helsinki-NLP/opus-mt-en-ml'
        }
        self.tokenizers = {}
        self.models = {}
        # Map langdetect codes to model codes
        self.lang_map = {'hi': 'hi', 'ml': 'ml', 'en': 'en'}
        # Flag to check if models were loaded
        self.loaded = False

    def load_translation_models(self):
        """Load models and tokenizers once on startup."""
        try:
            logger.info("Loading Multilingual Translation models...")
            # Load models and tokenizers once
            for k, v in self.model_name.items():
                self.tokenizers[k] = MarianTokenizer.from_pretrained(v)
                self.models[k] = MarianMTModel.from_pretrained(v)
            self.loaded = True
            logger.info("Multilingual Translation models loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading MarianMT models: {e}. Multilingual support disabled.")
            self.loaded = False

    def translate(self, text, src_lang, tgt_lang):
        """Performs translation."""
        if not self.loaded or src_lang == tgt_lang:
            return text  # No translation needed or models not loaded
        
        model_key = f"{src_lang}-{tgt_lang}"
        if model_key not in self.models:
            logger.warning(f"Translation model {model_key} not found. Returning original text.")
            return text
            
        tokenizer = self.tokenizers[model_key]
        model = self.models[model_key]
        
        # Move inputs to CPU/GPU/MPS if needed (Marian models are typically small enough for CPU)
        input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True)
        
        with torch.no_grad():
            translated = model.generate(input_ids)
        
        result = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
        return result[0]

    def process_user_input(self, user_input, model_response_func):
        """
        Translates input, calls the English-only model function, and translates back.
        `model_response_func` must accept an English query (str) and return (answer_en, confidence, used_model).
        """
        if not self.loaded:
            # If translator is not loaded, just call the model function directly
            answer_en, confidence, used_model = model_response_func(user_input)
            return answer_en, confidence, used_model, 'en' # Assume English for simplicity if translation failed

        # Step 1: Detect language (using langdetect)
        try:
            lang_detected = detect(user_input)
            src_lang = self.lang_map.get(lang_detected, 'en') # Default to 'en'
            
            # Use 'en' if detected language is outside supported local languages
            if src_lang not in ['hi', 'ml', 'en']:
                 src_lang = 'en'
        except Exception:
            # Fallback if langdetect fails
            src_lang = 'en'
        
        # Step 2: Translate input to English if needed
        if src_lang != 'en':
            user_input_en = self.translate(user_input, src_lang, 'en')
        else:
            user_input_en = user_input

        # Step 3: Query your English-trained model/dataset
        # The model_response_func is now expected to return (answer_en, confidence, used_model)
        answer_en, confidence, used_model = model_response_func(user_input_en)

        # Step 4: Translate answer back to user's language
        if src_lang != 'en':
            answer_local = self.translate(answer_en, 'en', src_lang)
        else:
            answer_local = answer_en

        return answer_local, confidence, used_model, src_lang

# ==================== END MULTILINGUAL TRANSLATOR CLASS ====================


# ==================== MODEL CONFIG ====================
# ... (Keep MODEL_MAP as is) ...
MODEL_MAP = {
    "insect_classifier": {
        "path": BASE_MODEL_PATH / "insect_image_model_outputs" / "insect_classifier_model.keras",
        "type": "keras",
        "task": "insect_classification",
        "labels_path": BASE_MODEL_PATH / "insect_image_model_outputs" / "class_indices.sav",
        "labels": []  # will be filled if labels_path exists
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

    # Text-to-Text Models (seq2seq)
    "farmer_advisory": {
        "path": BASE_MODEL_PATH / "farmer_call_query",
        "type": "huggingface_seq2seq",
        "task": "advisory",
        "config_files": [
            "config.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "special_tokens_map.json"
        ]
    },

    # FAQ: YOUR farmer_faq is a T5-like Seq2Seq model (T5ForConditionalGeneration)
    "faq_retrieval": {
        "path": BASE_MODEL_PATH / "farmer_faq",
        "type": "huggingface_seq2seq",   # load as seq2seq (T5)
        "task": "faq"
    },

    # Pesticide Models
    "pesticide_recommendation": {
        "path": BASE_MODEL_PATH / "pesticide_solution2.sav",
        "type": "sklearn",
        "task": "pesticide_recommendation",
        "labels": []
    },

    # Other ML Models
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

# Confidence thresholds
CONFIDENCE_THRESHOLD = 0.45
ESCALATION_WEBHOOK_URL = os.getenv("ESCALATION_WEBHOOK_URL", "https://example.com/webhook")

# Rate limiting
RATE_LIMIT_REQUESTS = 100
RATE_LIMIT_WINDOW = 60  # seconds

# FAQ Database (can be moved to MongoDB). Kept small for fallback testing.
FAQ_DATABASE = [
    {"question": "How to control pests in rice?", "answer": "Use integrated pest management..."},
    {"question": "Best fertilizer for wheat?", "answer": "NPK 20:20:20 is recommended..."},
]

# ==================== PYDANTIC MODELS ====================
# ... (Keep Pydantic Models as is) ...
class ContextData(BaseModel):
    farmer_id: Optional[str] = ""
    location: Optional[str] = ""
    crop: Optional[str] = ""
    season: Optional[str] = ""

class TextQueryRequest(BaseModel):
    query: str = Field(..., description="User query in Malayalam, Hindi, or English")
    context: Optional[ContextData] = None
    request_escalation: bool = False

class FeedbackRequest(BaseModel):
    query_id: str
    rating: int = Field(..., ge=1, le=5)
    feedback_text: Optional[str] = None

class CorrectionRequest(BaseModel):
    query_id: str
    correct_label: str
    notes: Optional[str] = None

# Renamed model_used -> used_model to avoid pydantic protected namespace warning
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

# ==================== MODEL LOADER ====================
# ... (Keep ModelLoader as is) ...
class ModelLoader:
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}

    def load_keras_model(self, model_path: Path):
        """Load Keras/TensorFlow model"""
        try:
            # use compile=False to avoid issues with custom objects if any
            model = tf.keras.models.load_model(str(model_path), compile=False)
            logger.info(f"Loaded Keras model: {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading Keras model {model_path}: {e}")
            return None

    def load_sklearn_model(self, model_path: Path):
        """Load scikit-learn model"""
        try:
            model = joblib.load(str(model_path))
            logger.info(f"Loaded sklearn model: {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading sklearn model {model_path}: {e}")
            return None

    def load_huggingface_model(self, model_path: Path):
        """Load HuggingFace seq2seq model and tokenizer (T5-like)"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path))
            # move model to available device (cpu or mps/cuda)
            if torch.cuda.is_available():
                model = model.to("cuda")
            elif torch.backends.mps.is_available():
                model = model.to("mps")
            logger.info(f"Loaded HuggingFace seq2seq model: {model_path}")
            return {"model": model, "tokenizer": tokenizer}
        except Exception as e:
            logger.error(f"Error loading HuggingFace model {model_path}: {e}")
            return None

    def load_sentence_transformer(self, model_path: Path):
        """Load Sentence Transformer model (if you provide one)"""
        try:
            model = SentenceTransformer(str(model_path))
            logger.info(f"Loaded Sentence Transformer: {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading Sentence Transformer {model_path}: {e}")
            return None

    def load_classical_pesticide(self, model_path: Path, files: Dict):
        """Load classical pesticide recommendation models"""
        try:
            models = {}
            for key, filename in files.items():
                file_path = model_path / filename
                with open(file_path, 'rb') as f:
                    models[key] = pickle.load(f)
            logger.info(f"Loaded classical pesticide models from: {model_path}")
            return models
        except Exception as e:
            logger.error(f"Error loading classical pesticide models {model_path}: {e}")
            return None

    def load_all_models(self):
        """Load all models defined in MODEL_MAP"""
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
                    obj = self.load_huggingface_model(model_path)
                    self.models[model_name] = obj
                    # If this is the insect classifier labels path is separate (handled below)
                elif model_type == "sentence_transformer":
                    self.models[model_name] = self.load_sentence_transformer(model_path)

                elif model_type == "classical_pesticide":
                    self.models[model_name] = self.load_classical_pesticide(
                        model_path, config["files"]
                    )

                # Post-load special handling: (e.g., load label mapping for insect_classifier)
                if model_name == "insect_classifier":
                    labels_path = config.get("labels_path")
                    if labels_path and labels_path.exists():
                        try:
                            with open(labels_path, "rb") as f:
                                idx_map = pickle.load(f)
                                # If idx_map is dict mapping class->index or index->class
                                if isinstance(idx_map, dict):
                                    # if keys are label names and values are indices -> take keys
                                    if all(isinstance(k, str) for k in idx_map.keys()):
                                        config["labels"] = list(idx_map.keys())
                                    else:
                                        # otherwise assume inverse mapping
                                        # create label list by sorting keys by value
                                        sorted_items = sorted(idx_map.items(), key=lambda kv: kv[1])
                                        config["labels"] = [k for k, v in sorted_items]
                                else:
                                    # fallback: just convert to list
                                    config["labels"] = list(idx_map)
                                logger.info(f"Loaded labels for insect_classifier: {config['labels']}")
                        except Exception as e:
                            logger.error(f"Failed to load insect labels from {labels_path}: {e}")

            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")

        logger.info(f"Loaded {len(self.models)} models successfully")

    def get_model(self, model_name: str):
        """Get loaded model by name"""
        return self.models.get(model_name)

# ==================== RATE LIMITER ====================
# ... (Keep RateLimiter as is) ...
class RateLimiter:
    def __init__(self, requests: int, window: int):
        self.requests = requests
        self.window = window
        self.clients = defaultdict(list)

    def is_allowed(self, client_id: str) -> bool:
        now = time.time()
        self.clients[client_id] = [
            t for t in self.clients[client_id] if now - t < self.window
        ]
        if len(self.clients[client_id]) >= self.requests:
            return False
        self.clients[client_id].append(now)
        return True

# ==================== FASTAPI APP ====================
app = FastAPI(title="Farmer Advisory System API", description="AI-powered agricultural advisory system", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
model_loader = ModelLoader()
rate_limiter = RateLimiter(RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW)
translator = MultilingualTranslator() # ADDED TRANSLATOR INSTANCE
db_client = None
db = None

# ==================== DATABASE EVENTS ====================
@app.on_event("startup")
async def startup_db_client():
    global db_client, db
    db_client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URL)
    db = db_client[DB_NAME]
    logger.info("Connected to MongoDB")
    # Load models (this will attempt to load all configured models)
    model_loader.load_all_models()
    # Load translation models
    translator.load_translation_models() # LOAD TRANSLATION MODELS HERE
    logger.info("All models loaded")

@app.on_event("shutdown")
async def shutdown_db_client():
    if db_client:
        db_client.close()
        logger.info("Closed MongoDB connection")

# ==================== RATE LIMIT MIDDLEWARE ====================
# ... (Keep rate_limit_middleware as is) ...
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware"""
    client_ip = request.client.host if request.client else "unknown"

    # Skip rate limiting for health check
    if request.url.path == "/health":
        return await call_next(request)

    if not rate_limiter.is_allowed(client_ip):
        return JSONResponse(
            status_code=429,
            content={"detail": "Too many requests. Please try again later."}
        )

    response = await call_next(request)
    return response

# ==================== HELPER FUNCTIONS ====================
# Removed the custom detect_language function since MultilingualTranslator handles it now.

# Renamed and restructured the main response logic into a function suitable for the translator.
def get_model_response_en(query_en: str, context: Optional[ContextData]) -> tuple[str, float, str]:
    """
    Routes English query to the appropriate model and returns (answer_en, confidence, used_model).
    This function is passed to translator.process_user_input.
    """
    answer_en = ""
    confidence = 0.0
    used_model = "fallback"

    # Route to appropriate model
    if should_use_pesticide_model(query_en):
        # Pesticide query - try LLM/Fallback (classical pesticide model is hard to adapt for arbitrary text)
        # Using LLM approach since classical model expects structured input (pest_name, crop)
        advisory_model = model_loader.get_model("farmer_advisory") 
        if advisory_model:
            answer_en, confidence = predict_pesticide_llm(query_en, advisory_model, context)
            used_model = "farmer_advisory (pesticide)"

    elif should_use_faq(query_en):
        # FAQ query
        faq_model = model_loader.get_model("faq_retrieval")
        if faq_model:
            answer_en, confidence = get_faq_answer(query_en, faq_model)
            used_model = "faq_retrieval" if answer_en else used_model

        # Fallback to advisory if FAQ didn't produce a good result
        if not answer_en or confidence < 0.4:
            advisory_model = model_loader.get_model("farmer_advisory")
            if advisory_model:
                answer_en, confidence = generate_advisory_response(query_en, advisory_model, context)
                used_model = "farmer_advisory (faq_fallback)"

    else:
        # General advisory query
        advisory_model = model_loader.get_model("farmer_advisory")
        if advisory_model:
            answer_en, confidence = generate_advisory_response(query_en, advisory_model, context)
            used_model = "farmer_advisory"
    
    # Final fallback if all else fails
    if not answer_en:
        answer_en = "I'm sorry, I couldn't find a relevant answer. Please try rephrasing your question or request a manual escalation."
        confidence = 0.05
        used_model = "final_fallback"

    return answer_en, confidence, used_model


def should_use_pesticide_model(query: str) -> bool:
    """Check if query is about pesticides"""
    pesticide_keywords = [
        "pesticide", "pest", "control", "spray", "insect", "disease",
        "fungicide", "herbicide", "aphid", "beetle", "caterpillar", "cure" # Added "cure"
    ]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in pesticide_keywords)

def should_use_faq(query: str) -> bool:
    """Check if query should use FAQ retrieval"""
    faq_keywords = ["how", "what", "when", "where", "why", "best", "recommend", "advice"] # Added "recommend", "advice"
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in faq_keywords) and len(query.split()) < 25

async def create_escalation(query_id: str, query: str, confidence: float,
                           context: Optional[ContextData]) -> str:
    """Create escalation record"""
    escalation_id = str(uuid.uuid4())

    escalation_data = {
        "escalation_id": escalation_id,
        "query_id": query_id,
        "query": query,
        "confidence": confidence,
        "context": context.dict() if context else {},
        "status": "pending",
        "created_at": datetime.utcnow().isoformat(),
        "officer_summary": f"Low confidence query ({confidence:.2f}). Manual review required."
    }

    # store in DB if available
    if db:
        await db.escalations.insert_one(escalation_data)

    # TODO: optionally POST to webhook ESCALATION_WEBHOOK_URL
    logger.info(f"Created escalation: {escalation_id}")
    return escalation_id

def preprocess_image(image: Image.Image, target_size=(224, 224)) -> np.ndarray:
    """Preprocess image for model prediction"""
    image = image.convert('RGB')
    image = image.resize(target_size)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1D numpy vectors"""
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
    return float(np.dot(a, b) / denom)

def generate_faq_answer_seq2seq(question: str, model_data: Dict) -> tuple:
    """Use T5-like seq2seq model to generate an answer and a heuristic confidence."""
    tokenizer = model_data["tokenizer"]
    model = model_data["model"]
    # Optionally prefix to give T5 some task hint; your model may expect plain text
    prefixed = f"question: {question}"
    inputs = tokenizer(prefixed, return_tensors="pt", truncation=True, max_length=512)
    # move inputs to model device if needed
    device = next(model.parameters()).device if hasattr(model, "parameters") else torch.device("cpu")
    for k, v in inputs.items():
        inputs[k] = v.to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=200,
            num_beams=4,
            early_stopping=True
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Heuristic confidence: longer and not-empty answers get higher score
    confidence = min(0.95, 0.35 + len(decoded.split()) / 80.0)
    return decoded, confidence

def get_faq_answer(query: str, model) -> tuple:
    """
    Unified FAQ answer function.
    - If `model` is a HuggingFace seq2seq (dict with 'model' and 'tokenizer'), generate directly.
    - If `model` is a SentenceTransformer, compute similarity against static FAQ_DATABASE.
    """
    if not model:
        return None, 0.0

    # Seq2seq model (HuggingFace) is stored as dict {"model":..., "tokenizer":...}
    if isinstance(model, dict) and "model" in model and "tokenizer" in model:
        try:
            return generate_faq_answer_seq2seq(query, model)
        except Exception as e:
            logger.error(f"Error generating FAQ answer from seq2seq model: {e}")
            return None, 0.0

    # SentenceTransformer fallback: use similarity to FAQ_DATABASE
    try:
        # Note: If you don't use SentenceTransformer, this block should be removed.
        # For now, keeping it commented out for cleaner execution flow unless you have that model.
        # query_embedding = model.encode([query])[0]
        # best_match = None
        # best_score = 0.0
        # for faq in FAQ_DATABASE:
        #     faq_embedding = model.encode([faq["question"]])[0]
        #     sim = cosine_similarity(np.array(query_embedding), np.array(faq_embedding))
        #     if sim > best_score:
        #         best_score = sim
        #         best_match = faq
        # if best_match and best_score > 0.55:
        #     return best_match["answer"], float(best_score)
        # else:
            return None, 0.0
    except Exception as e:
        logger.error(f"Error using sentence-transformer FAQ model: {e}")
    return None, 0.0

def generate_advisory_response(query: str, model_data: Dict, context: Optional[ContextData]) -> tuple:
    """Generate advisory response using LLM (seq2seq)"""
    if not model_data or not isinstance(model_data, dict):
        return "Advisory model not available", 0.0
    model = model_data["model"]
    tokenizer = model_data["tokenizer"]

    # Build context-aware prompt
    context_str = ""
    if context:
        context_parts = []
        if context.crop:
            context_parts.append(f"Crop: {context.crop}")
        if context.location:
            context_parts.append(f"Location: {context.location}")
        if context.season:
            context_parts.append(f"Season: {context.season}")
        if context_parts:
            context_str = f"Context: {', '.join(context_parts)}. "

    full_query = f"{context_str}Query: {query}"
    inputs = tokenizer(full_query, return_tensors="pt", truncation=True, max_length=512)
    device = next(model.parameters()).device if hasattr(model, "parameters") else torch.device("cpu")
    for k, v in inputs.items():
        inputs[k] = v.to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=256,
            num_beams=4,
            early_stopping=True,
            temperature=0.7
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    confidence = min(0.9, 0.5 + len(response.split()) / 100)
    return response, confidence

def predict_pesticide_classical(pest_name: str, crop: str, models: Dict) -> tuple:
    """Predict pesticide using classical RF model"""
    # ... (Keep classical prediction as is, note its limitation for arbitrary queries)
    # NOTE: This function is not used in the new query flow for simplicity/robustness against arbitrary user text
    try:
        rf_model = models["model"]
        label_encoder = models["label_encoder"]
        onehot_encoder = models["onehot_encoder"]

        # Prepare input (adjust based on your training features)
        # NOTE: this is placeholder — adapt to your pipeline's features
        input_data = np.array([[pest_name, crop]])
        encoded = onehot_encoder.transform(input_data)

        prediction = rf_model.predict(encoded)
        probabilities = rf_model.predict_proba(encoded)

        pesticide = label_encoder.inverse_transform(prediction)[0]
        confidence = float(np.max(probabilities))

        return pesticide, confidence

    except Exception as e:
        logger.error(f"Error in classical pesticide prediction: {e}")
        return "Unable to predict", 0.0

def predict_pesticide_llm(query: str, model_data: Dict, context: Optional[ContextData]) -> tuple:
    """Predict pesticide using LLM (repurposing advisory model)"""
    if not model_data:
        return "LLM not available", 0.0
    model = model_data["model"]
    tokenizer = model_data["tokenizer"]

    context_str = ""
    if context and context.crop:
        context_str = f"for {context.crop} crop "

    # Better prompt for advisory model to give a pesticide recommendation
    prompt = f"Recommend a suitable pesticide or treatment {context_str}for the following agricultural issue: {query}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    device = next(model.parameters()).device if hasattr(model, "parameters") else torch.device("cpu")
    for k, v in inputs.items():
        inputs[k] = v.to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=128,
            num_beams=3,
            early_stopping=True
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    confidence = 0.75 # Defaulting to a high confidence for LLM recommendation
    return response, confidence

# ==================== API ENDPOINTS ====================
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "models_loaded": len(model_loader.models),
        "translator_loaded": translator.loaded # Include translator status
    }

@app.post("/api/query", response_model=QueryResponse)
async def process_text_query(request: TextQueryRequest):
    """
    Process text query with intelligent routing and multilingual support
    Supports Malayalam, Hindi, and English.
    """
    query_id = str(uuid.uuid4())
    original_query = request.query

    try:
        # 1. Use the MultilingualTranslator to handle the entire request flow
        # It detects language, translates to English, calls the core model function, and translates back.
        
        # Define the function that the translator will call with the English query
        model_func = lambda q_en: get_model_response_en(q_en, request.context)
        
        # Get final local answer, confidence, used model, and detected language
        answer_local, confidence, used_model, detected_lang_code = \
            translator.process_user_input(original_query, model_func)
            
        answer = answer_local # The final response in the user's language
        language = detected_lang_code # 'hi', 'ml', or 'en'
        
        # 2. Check for escalation
        escalation_id = None
        if confidence < CONFIDENCE_THRESHOLD or request.request_escalation:
            # We escalate with the ORIGINAL query
            escalation_id = await create_escalation(query_id, original_query, confidence, request.context)

        # 3. Store query in database
        query_data = {
            "query_id": query_id,
            "query": original_query,
            "language": language,
            "answer": answer,
            "confidence": confidence,
            "used_model": used_model,
            "context": request.context.dict() if request.context else {},
            "escalation_id": escalation_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        if db:
            await db.queries.insert_one(query_data)

        # 4. Return response
        return QueryResponse(
            query_id=query_id,
            answer=answer,
            confidence=confidence,
            used_model=used_model,
            timestamp=query_data["timestamp"],
            escalation_id=escalation_id
        )

    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        # Log the fallback for multilingual issues
        logger.warning("Falling back to English-only processing due to an error.")
        
        # Fallback to English-only flow if multilingual part failed
        try:
            answer, confidence, used_model = get_model_response_en(original_query, request.context)
            language = 'en' # Assuming English on fallback

            # Store fallback query in database
            query_data = {
                "query_id": query_id,
                "query": original_query,
                "language": language,
                "answer": answer,
                "confidence": confidence,
                "used_model": used_model + "_fallback",
                "context": request.context.dict() if request.context else {},
                "escalation_id": None, # Skip escalation check on exception fallback
                "timestamp": datetime.utcnow().isoformat()
            }
            if db:
                await db.queries.insert_one(query_data)

            return QueryResponse(
                query_id=query_id,
                answer=answer,
                confidence=confidence,
                used_model=query_data["used_model"],
                timestamp=query_data["timestamp"],
                escalation_id=None
            )
        except Exception as fallback_e:
            logger.error(f"Error during English fallback: {fallback_e}", exc_info=True)
            raise HTTPException(status_code=500, detail="A critical error occurred while processing the query.")
            
# ... (Keep other endpoints as is: /api/image, /api/feedback, /api/correction, /api/reload-models, /api/escalations, /api/stats) ...
@app.post("/api/image", response_model=ImageResponse)
async def process_image(
    file: UploadFile = File(...),
    farmer_id: Optional[str] = None,
    location: Optional[str] = None,
    crop: Optional[str] = None,
    season: Optional[str] = None,
    request_escalation: bool = False
):
    """
    Process image upload for pest/disease/crop classification
    """
    query_id = str(uuid.uuid4())

    try:
        # Read and preprocess image
        image_data = await file.read()
        image = Image.open(BytesIO(image_data))

        # Try multiple models and get best prediction
        predictions = []

        # Try insect classifier
        insect_model = model_loader.get_model("insect_classifier")
        if insect_model:
            img_array = preprocess_image(image, target_size=(224, 224))
            pred = insect_model.predict(img_array)
            class_idx = int(np.argmax(pred[0]))
            confidence = float(pred[0][class_idx])

            labels = MODEL_MAP["insect_classifier"].get("labels", [])
            if class_idx < len(labels):
                predictions.append({
                    "label": labels[class_idx],
                    "confidence": confidence,
                    "model": "insect_classifier"
                })

        # Try disease classifier
        disease_model = model_loader.get_model("disease_classifier")
        if disease_model:
            img_array = preprocess_image(image, target_size=(224, 224))
            pred = disease_model.predict(img_array)
            class_idx = int(np.argmax(pred[0]))
            confidence = float(pred[0][class_idx])

            labels = MODEL_MAP["disease_classifier"]["labels"]
            if class_idx < len(labels):
                predictions.append({
                    "label": labels[class_idx],
                    "confidence": confidence,
                    "model": "disease_classifier"
                })

        # Get best prediction
        if predictions:
            best = max(predictions, key=lambda x: x["confidence"])
            label = best["label"]
            confidence = best["confidence"]
            used_model = best["model"]
        else:
            raise HTTPException(status_code=500, detail="No models available")

        # Check for escalation
        escalation_id = None
        if confidence < CONFIDENCE_THRESHOLD or request_escalation:
            context = ContextData(
                farmer_id=farmer_id or "",
                location=location or "",
                crop=crop or "",
                season=season or ""
            )
            escalation_id = await create_escalation(query_id, f"Image classification: {label}", confidence, context)

        # Store in database
        image_data_doc = {
            "query_id": query_id,
            "label": label,
            "confidence": confidence,
            "used_model": used_model,
            "context": {
                "farmer_id": farmer_id or "",
                "location": location or "",
                "crop": crop or "",
                "season": season or ""
            },
            "escalation_id": escalation_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        if db:
            await db.queries.insert_one(image_data_doc)

        return ImageResponse(
            query_id=query_id,
            label=label,
            confidence=confidence,
            used_model=used_model,
            timestamp=image_data_doc["timestamp"],
            escalation_id=escalation_id
        )

    except Exception as e:
        logger.error(f"Error processing image: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """Submit user feedback for a query"""
    try:
        feedback_data = {
            "feedback_id": str(uuid.uuid4()),
            "query_id": feedback.query_id,
            "rating": feedback.rating,
            "feedback_text": feedback.feedback_text,
            "timestamp": datetime.utcnow().isoformat()
        }

        if db:
            await db.feedback.insert_one(feedback_data)

        return {"status": "success", "message": "Feedback recorded"}

    except Exception as e:
        logger.error(f"Error submitting feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/correction")
async def submit_correction(correction: CorrectionRequest):
    """Submit corrected label for model improvement"""
    try:
        correction_data = {
            "correction_id": str(uuid.uuid4()),
            "query_id": correction.query_id,
            "correct_label": correction.correct_label,
            "notes": correction.notes,
            "timestamp": datetime.utcnow().isoformat()
        }

        if db:
            await db.corrections.insert_one(correction_data)

        return {"status": "success", "message": "Correction recorded"}

    except Exception as e:
        logger.error(f"Error submitting correction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/reload-models")
async def reload_models():
    """Hot reload all models without restarting server"""
    try:
        model_loader.models.clear()
        model_loader.tokenizers.clear()
        model_loader.load_all_models()
        # Reload translation models as well
        translator.tokenizers.clear()
        translator.models.clear()
        translator.load_translation_models()

        return {
            "status": "success",
            "message": f"Reloaded {len(model_loader.models)} models and translation models"
        }

    except Exception as e:
        logger.error(f"Error reloading models: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/escalations")
async def get_escalations(status: Optional[str] = None):
    """Get escalated queries (for officer dashboard)"""
    try:
        query = {}
        if status:
            query["status"] = status

        if db:
            escalations = await db.escalations.find(query).to_list(length=100)
            for esc in escalations:
                esc["_id"] = str(esc["_id"])
        else:
            escalations = []

        return {"escalations": escalations}

    except Exception as e:
        logger.error(f"Error fetching escalations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    try:
        if db:
            total_queries = await db.queries.count_documents({})
            total_escalations = await db.escalations.count_documents({})
            pending_escalations = await db.escalations.count_documents({"status": "pending"})
            total_feedback = await db.feedback.count_documents({})
        else:
            total_queries = total_escalations = pending_escalations = total_feedback = 0

        return {
            "total_queries": total_queries,
            "total_escalations": total_escalations,
            "pending_escalations": pending_escalations,
            "total_feedback": total_feedback,
            "models_loaded": len(model_loader.models),
            "translator_loaded": translator.loaded
        }

    except Exception as e:
        logger.error(f"Error fetching stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ==================== RUN SERVER ====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)