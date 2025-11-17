# main.py
# FastAPI backend for Farmer Advisory System
# Complete updated version with all fixes and improvements

# ==================== IMPORTS ====================
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
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
            with torch.no_grad():
                translated = model.generate(input_ids, max_length=512)
            return tokenizer.decode(translated[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return text

    def process_user_input(self, user_input: str, model_response_func):
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
    
    if db:
        try:
            escalation_doc = {
                "escalation_id": escalation_id,
                "query_id": query_id,
                "query": query,
                "confidence": confidence,
                "context": context.dict() if context else {},
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
    """Get response from appropriate model based on query"""
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
            tokenizer = advisory_model["tokenizer"]
            model = advisory_model["model"]
            
            # Simple retrieval using BERT embeddings
            inputs = tokenizer(
                query_en,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            # This is a placeholder - actual implementation depends on your model
            answer = "Based on your query, I recommend consulting with a local agricultural expert for specific advice tailored to your situation."
            confidence = 0.65
            
            return answer, confidence, "farmer_advisory"
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

@app.post("/api/query", response_model=QueryResponse)
async def process_text_query(request: TextQueryRequest):
    """Process text-based agricultural queries with multilingual support"""
    query_id = str(uuid.uuid4())
    
    try:
        # Process query through translator and model
        if ENABLE_MULTILINGUAL:
            answer_local, confidence, used_model, lang = translator.process_user_input(
                request.query,
                get_model_response
            )
        else:
            answer_local, confidence, used_model = get_model_response(request.query)
            lang = 'en'
        
        # Create escalation if confidence is low
        escalation_id = None
        if confidence < CONFIDENCE_THRESHOLD or request.request_escalation:
            escalation_id = await create_escalation(
                query_id,
                request.query,
                confidence,
                request.context
            )
        
        # Store query in database
        doc = {
            "query_id": query_id,
            "query": request.query,
            "answer": answer_local,
            "confidence": confidence,
            "used_model": used_model,
            "detected_language": lang,
            "escalation_id": escalation_id,
            "context": request.context.dict() if request.context else {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if db:
            try:
                await db.queries.insert_one(doc)
            except Exception as e:
                logger.error(f"Database insert error: {e}")
        
        return QueryResponse(
            query_id=query_id,
            answer=answer_local,
            confidence=confidence,
            used_model=used_model,
            timestamp=doc["timestamp"],
            escalation_id=escalation_id
        )
    
    except Exception as e:
        logger.error(f"Query processing error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your query. Please try again."
        )

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
    """Retrieve a specific query by ID"""
    if not db:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
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
    """List recent queries with optional filtering"""
    if not db:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        filter_query = {}
        if farmer_id:
            filter_query["context.farmer_id"] = farmer_id
        
        cursor = db.queries.find(filter_query, {"_id": 0}).sort("timestamp", -1).skip(skip).limit(limit)
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
            "total_queries": await db.queries.count_documents({}),
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
        result = await db.queries.aggregate(pipeline).to_list(1)
        stats["average_confidence"] = result[0]["avg_confidence"] if result else 0
        
        # Model usage distribution
        model_usage_pipeline = [
            {"$group": {"_id": "$used_model", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        model_usage = await db.queries.aggregate(model_usage_pipeline).to_list(10)
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
            "query": "/api/query",
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