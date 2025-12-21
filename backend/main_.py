from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List, Tuple
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
from sentence_transformers import SentenceTransformer
import faiss
import torch
from collections import defaultdict
from langdetect import detect
from dotenv import load_dotenv
from asyncio import Lock
import json
import pandas as pd
from difflib import get_close_matches
import re

# ==================== ENVIRONMENT & LOGGING SETUP ====================
load_dotenv()
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
if not logger.handlers:
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(sh)
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

INSTRUCTION_PREFIX = "instruction:"
def instruction_tag(action: str) -> str:
    return f"{INSTRUCTION_PREFIX}{action}"

# ==================== STABILITY / CONFIG ====================
ENABLE_GPU = os.getenv("ENABLE_GPU", "false").lower() == "true"
if not ENABLE_GPU:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    logger.info("Deep Learning models are configured to run on CPU for stability.")

MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "farmer_advisory_system")
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", 100))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", 60))
BASE_DIR = Path(__file__).resolve().parent
MODEL_BASE_PATH_STR = os.getenv("MODEL_BASE_PATH", "./models")
BASE_MODEL_PATH = BASE_DIR / MODEL_BASE_PATH_STR if MODEL_BASE_PATH_STR.startswith(".") else Path(MODEL_BASE_PATH_STR)
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE_MB", "10")) * 1024 * 1024
ALLOWED_EXTENSIONS_STR = os.getenv("ALLOWED_IMAGE_EXTENSIONS", ".jpg,.jpeg,.png,.webp")
ALLOWED_EXTENSIONS = set(ext.strip() for ext in ALLOWED_EXTENSIONS_STR.split(","))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.45"))
ALLOWED_ORIGINS_STR = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173")
ALLOWED_ORIGINS = [origin.strip() for origin in ALLOWED_ORIGINS_STR.split(",")]

# ==================== TRANSLATOR (placeholder) ====================
class MultilingualTranslator:
    def __init__(self):
        self.loaded = False

    def load_translation_models(self):
        logger.info("Multilingual Translation models marked as ready (using placeholder logic).")
        self.loaded = True

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        if not self.loaded or src_lang == tgt_lang:
            return text
        if tgt_lang == 'en' and src_lang != 'en':
            return text.replace("खाद", "fertilizer").replace("चावल", "rice").replace("നെല്ല്", "rice")
        if src_lang == 'en' and tgt_lang != 'en':
            return f"[{tgt_lang.upper()} Translation: {text}]"
        return text

# ==================== DATA STORES ====================
class DataStore:
    """Central data store for all agricultural datasets"""
    def __init__(self):
        self.crop_production_data: List[Dict] = []
        self.crop_calendar_data: List[Dict] = []
        self.call_query_data: List[Dict] = []
        self.pesticide_recommendations: Dict[str, List[str]] = {}
        self.pest_solutions: List[Dict] = []
        self.faq_data: List[Dict] = []
        
    def load_crop_production(self, file_path: Path):
        """Load crop production dataset"""
        try:
            if file_path.suffix == '.csv':
                df = pd.read_csv(file_path)
            elif file_path.suffix in ['.pkl', '.pickle']:
                df = pd.read_pickle(file_path)
            else:
                logger.warning(f"Unsupported file format for crop production: {file_path}")
                return
            
            self.crop_production_data = df.to_dict('records')
            logger.info(f"Loaded {len(self.crop_production_data)} crop production records")
        except Exception as e:
            logger.exception(f"Failed to load crop production data: {e}")
    
    def load_crop_calendar(self, file_path: Path):
        """Load crop calendar dataset"""
        try:
            if file_path.suffix == '.csv':
                df = pd.read_csv(file_path)
            elif file_path.suffix in ['.pkl', '.pickle']:
                df = pd.read_pickle(file_path)
            else:
                logger.warning(f"Unsupported file format for crop calendar: {file_path}")
                return
            
            self.crop_calendar_data = df.to_dict('records')
            logger.info(f"Loaded {len(self.crop_calendar_data)} crop calendar records")
        except Exception as e:
            logger.exception(f"Failed to load crop calendar data: {e}")
    
    def load_call_query(self, file_path: Path):
        """Load agricultural Q&A dataset"""
        try:
            if file_path.suffix == '.csv':
                df = pd.read_csv(file_path)
            elif file_path.suffix in ['.pkl', '.pickle']:
                df = pd.read_pickle(file_path)
            else:
                logger.warning(f"Unsupported file format for call query: {file_path}")
                return
            
            # Ensure columns are named 'questions' and 'answers'
            if 'questions' in df.columns and 'answers' in df.columns:
                self.call_query_data = df.to_dict('records')
                logger.info(f"Loaded {len(self.call_query_data)} call query records")
            else:
                logger.warning(f"Call query data missing 'questions' or 'answers' columns")
        except Exception as e:
            logger.exception(f"Failed to load call query data: {e}")
    
    def load_pesticide_recommendations(self, file_path: Path):
        """Load pesticide recommendations (Pest Name -> Pesticides mapping)"""
        try:
            if file_path.suffix == '.csv':
                df = pd.read_csv(file_path)
            elif file_path.suffix in ['.pkl', '.pickle']:
                df = pd.read_pickle(file_path)
            elif file_path.suffix == '.json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    self.pesticide_recommendations = data
                    logger.info(f"Loaded {len(self.pesticide_recommendations)} pesticide recommendations")
                    return
            else:
                logger.warning(f"Unsupported file format for pesticide recommendations: {file_path}")
                return
            
            # Convert DataFrame to dict mapping
            if 'Pest Name' in df.columns and 'Most Commonly Used Pesticides' in df.columns:
                for _, row in df.iterrows():
                    pest_name = str(row['Pest Name']).strip().lower()
                    pesticides = str(row['Most Commonly Used Pesticides']).split(',')
                    self.pesticide_recommendations[pest_name] = [p.strip() for p in pesticides]
                logger.info(f"Loaded {len(self.pesticide_recommendations)} pesticide recommendations")
        except Exception as e:
            logger.exception(f"Failed to load pesticide recommendations: {e}")
    
    def load_pest_solutions(self, file_path: Path):
        """Load detailed pest solutions dataset"""
        try:
            if file_path.suffix == '.csv':
                df = pd.read_csv(file_path)
            elif file_path.suffix in ['.pkl', '.pickle']:
                df = pd.read_pickle(file_path)
            else:
                logger.warning(f"Unsupported file format for pest solutions: {file_path}")
                return
            
            self.pest_solutions = df.to_dict('records')
            logger.info(f"Loaded {len(self.pest_solutions)} pest solution records")
        except Exception as e:
            logger.exception(f"Failed to load pest solutions: {e}")
    
    def search_crop_production(self, state: str = None, crop: str = None, season: str = None) -> List[Dict]:
        """Search crop production data"""
        results = self.crop_production_data
        
        if state:
            state_lower = state.lower()
            results = [r for r in results if state_lower in str(r.get('State_Name', '')).lower()]
        
        if crop:
            crop_lower = crop.lower()
            results = [r for r in results if crop_lower in str(r.get('Crop', '')).lower()]
        
        if season:
            season_lower = season.lower()
            results = [r for r in results if season_lower in str(r.get('Season', '')).lower()]
        
        return results[:10]  # Limit to top 10 results
    
    def search_crop_calendar(self, state: str = None, crop: str = None) -> List[Dict]:
        """Search crop calendar data"""
        results = self.crop_calendar_data
        
        if state:
            state_lower = state.lower()
            results = [r for r in results if state_lower in str(r.get('States/Uts', '')).lower()]
        
        if crop:
            crop_lower = crop.lower()
            # Search in all crop-related columns
            results = [r for r in results if any(
                crop_lower in str(v).lower() 
                for k, v in r.items() 
                if k not in ['States/Uts', 'Period']
            )]
        
        return results[:5]
    
    def search_pesticide_by_pest(self, pest_name: str) -> Optional[List[str]]:
        """Get pesticide recommendations for a pest"""
        pest_lower = pest_name.lower()
        
        # Exact match
        if pest_lower in self.pesticide_recommendations:
            return self.pesticide_recommendations[pest_lower]
        
        # Fuzzy match
        matches = get_close_matches(pest_lower, self.pesticide_recommendations.keys(), n=1, cutoff=0.6)
        if matches:
            return self.pesticide_recommendations[matches[0]]
        
        return None
    
    def search_pest_solutions(self, pest_name: str = None, crop: str = None) -> List[Dict]:
        """Search detailed pest solutions"""
        results = self.pest_solutions
        
        if pest_name:
            pest_lower = pest_name.lower()
            results = [r for r in results if pest_lower in str(r.get('disease_or_pest', '')).lower()]
        
        if crop:
            crop_lower = crop.lower()
            results = [r for r in results if crop_lower in str(r.get('crop', '')).lower()]
        
        return results[:5]

# ==================== RAG Pipeline ====================
class RAGPipeline:
    def __init__(self, data_store: DataStore):
        self.embedder: Optional[SentenceTransformer] = None
        self.faiss_index: Optional[faiss.Index] = None
        self.faq_data: Optional[List[Dict[str, Any]]] = None
        self.loaded = False
        self.data_store = data_store

    def load(self, embedder_name: str, faiss_path: Path, data_path: Path):
        try:
            logger.info("Starting RAG Pipeline loading...")
            with open(data_path, 'rb') as f:
                loaded_data = pickle.load(f)

            if isinstance(loaded_data, pd.DataFrame):
                self.faq_data = loaded_data.to_dict('records')
            else:
                self.faq_data = loaded_data if isinstance(loaded_data, list) else [loaded_data]

            self.data_store.faq_data = self.faq_data

            self.faiss_index = faiss.read_index(str(faiss_path))
            device = 'cuda' if ENABLE_GPU and torch.cuda.is_available() else 'cpu'
            self.embedder = SentenceTransformer(embedder_name, device=device)

            self.loaded = True
            logger.info("RAG Pipeline loaded successfully.")
        except Exception as e:
            logger.exception("FATAL Error loading RAG pipeline: %s", e)
            self.loaded = False

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if not self.loaded:
            logger.debug("RAGPipeline.search called but pipeline not loaded.")
            return []
        try:
            query_embedding = np.array([self.embedder.encode(query, convert_to_tensor=False).astype('float32')])
            distances, indices = self.faiss_index.search(query_embedding, top_k)
            results = []
            max_dist = float(np.max(distances)) if np.array(distances).size else 0.0
            for dist, idx in zip(distances[0], indices[0]):
                if idx >= 0 and idx < len(self.faq_data):
                    confidence = float(max(0.0, 1.0 - (dist / max_dist))) if max_dist > 0 else 0.9
                    results.append({"data": self.faq_data[idx], "confidence": confidence, "distance": float(dist)})
            return results
        except Exception as e:
            logger.exception("RAG search error: %s", e)
            return []

    def generate_response(self, query: str, context: List[Dict[str, Any]]) -> Tuple[str, float]:
        if not context:
            return "I couldn't find a relevant answer in the knowledge base. Please try rephrasing.", 0.1
        top_result = context[0]
        answer = top_result.get('data', {}).get('answer') or top_result.get('data', {}).get('text') or str(top_result.get('data', {}))
        confidence = float(top_result.get('confidence', 0.0))
        if confidence < CONFIDENCE_THRESHOLD:
            return f"I found this answer, but my confidence is low ({confidence:.2f}): {answer}", confidence
        return answer, confidence
    
    def search_call_query(self, query: str) -> Tuple[str, float]:
        """Search call_query dataset for agricultural Q&A"""
        if not self.data_store.call_query_data:
            return None, 0.0
        
        query_lower = query.lower()
        best_match = None
        best_score = 0.0
        
        for record in self.data_store.call_query_data:
            question = str(record.get('questions', '')).lower()
            # Simple keyword matching
            common_words = set(query_lower.split()) & set(question.split())
            score = len(common_words) / max(len(query_lower.split()), 1)
            
            if score > best_score:
                best_score = score
                best_match = record
        
        if best_match and best_score > 0.3:
            answer = best_match.get('answers', 'No answer available')
            return answer, min(best_score * 2, 0.95)  # Scale confidence
        
        return None, 0.0

# ==================== MODEL LOADER ====================
class ModelLoader:
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.data_store = DataStore()
        self.rag_pipeline = RAGPipeline(self.data_store)
        self.label_maps: Dict[str, Dict[int, str]] = {}

    def load_keras_model(self, model_path: Path):
        try:
            return tf.keras.models.load_model(str(model_path))
        except Exception as e:
            logger.exception("Keras model load error for %s: %s", model_path, e)
            return None

    def load_sklearn_model(self, model_path: Path):
        try:
            suffix = model_path.suffix.lower()
            if suffix == '.pkl':
                with open(model_path, 'rb') as f:
                    return pickle.load(f)
            elif suffix in ['.joblib', '.jl']:
                return joblib.load(model_path)
            else:
                try:
                    return joblib.load(model_path)
                except Exception:
                    with open(model_path, 'rb') as f:
                        return pickle.load(f)
        except Exception as e:
            logger.exception("SKLearn/Joblib model load error for %s: %s", model_path, e)
            return None

    def load_all_models(self):
        """Load all models and datasets"""
        # Define model map
        MODEL_MAP = {
            # FAQ/RAG Model
            "faq_data": {"path": BASE_MODEL_PATH / "faqs_model" / "faq_data.pkl", "type": "pickle_data", "task": "rag_data"},
            "faq_index": {"path": BASE_MODEL_PATH / "faqs_model" / "faq_index.faiss", "type": "faiss", "task": "rag_index"},
            "faq_model_name": {"path": BASE_MODEL_PATH / "faqs_model" / "model_name.txt", "type": "text", "task": "rag_embedder_name"},
            
            # Insect Classifier
            "insect_classifier": {"path": BASE_MODEL_PATH / "insect_classifier" / "insect_classifier.keras", "type": "keras", "task": "insect_classification"},
            "insect_label_map": {"path": BASE_MODEL_PATH / "insect_classifier" / "label_map.json", "type": "json", "task": "insect_labels"},
            
            # Other ML models
            "fertilizer_label_map": {"path": BASE_MODEL_PATH / "fertilizer_model" / "labels.json", "type": "json", "task": "fertilizer_labels"},
            "agri_qa_model": {"path": BASE_MODEL_PATH / "agri_qa_model.pkl", "type": "pickle_sklearn", "task": "agri_qa"},
            "crop_calendar": {"path": BASE_MODEL_PATH / "crop_calendar.pkl", "type": "pickle_sklearn", "task": "crop_calendar"},
            "fertilizer_recommendation_model": {"path": BASE_MODEL_PATH / "fertilizer_recommendation_model.joblib", "type": "joblib_sklearn", "task": "fertilizer_recommendation"},
            "ipm_model": {"path": BASE_MODEL_PATH / "ipm_model.pkl", "type": "pickle_sklearn", "task": "ipm_recommendation"},
            "pest_recommendation_model": {"path": BASE_MODEL_PATH / "pest_recommendation_model.pkl", "type": "pickle_sklearn", "task": "pest_recommendation"},
        }
        
        faq_embedder_name = None
        faq_faiss_path = None
        faq_data_path = None

        for model_name, config in MODEL_MAP.items():
            path = config.get("path")
            if path is None:
                continue

            path = Path(path)
            if not path.exists():
                logger.warning("Model file not found: %s. Skipping %s.", path, model_name)
                continue

            logger.info("Attempting to load: %s (type=%s)", model_name, config.get("type"))

            try:
                typ = config.get("type", "")
                if typ == "keras":
                    self.models[model_name] = self.load_keras_model(path)
                elif typ in ["pickle_sklearn", "joblib_sklearn"]:
                    self.models[model_name] = self.load_sklearn_model(path)
                elif typ == "text" and model_name == "faq_model_name":
                    with open(path, 'r') as f:
                        faq_embedder_name = f.read().strip()
                elif typ == "faiss" and model_name == "faq_index":
                    faq_faiss_path = path
                elif typ == "pickle_data" and model_name == "faq_data":
                    faq_data_path = path
                elif typ == "json":
                    with open(path, 'r') as f:
                        label_map_str = json.load(f)
                        try:
                            self.label_maps[model_name] = {int(k): v for k, v in label_map_str.items()}
                        except Exception:
                            self.label_maps[model_name] = label_map_str
                    logger.info("Loaded JSON map %s with %d entries.", model_name, len(self.label_maps.get(model_name, {})))
            except Exception as e:
                logger.exception("Failed to process model %s: %s", model_name, e)

        # Load RAG pipeline
        if faq_embedder_name and faq_faiss_path and faq_data_path:
            self.rag_pipeline.load(faq_embedder_name, faq_faiss_path, faq_data_path)
            if self.rag_pipeline.loaded:
                self.models["rag_pipeline"] = self.rag_pipeline
        
        # Load datasets
        self._load_datasets()

        logger.info("ModelLoader finished. Loaded %d model entries and %d label maps.", len(self.models), len(self.label_maps))
    
    def _load_datasets(self):
        """Load all CSV/pickle datasets"""
        dataset_files = {
            "crop_production": BASE_MODEL_PATH / "crop_production.csv",
            "crop_calendar": BASE_MODEL_PATH / "crop_calendar.csv",
            "call_query": BASE_MODEL_PATH / "call_query.csv",
            "pesticide_recommendations": BASE_MODEL_PATH / "Pesticides_recomm.csv",  # Match your actual filename
            "pest_solutions": BASE_MODEL_PATH / "pest_solution.csv",
        }
        
        # Try alternate file paths if primary not found
        alternate_paths = {
            "crop_calendar": BASE_MODEL_PATH / "crop_calendar.pkl",
            "pesticide_recommendations": BASE_MODEL_PATH / "pesticides_recomm.csv",  # Lowercase alternative
        }
        
        for dataset_name, file_path in dataset_files.items():
            # Try primary path first
            if not file_path.exists() and dataset_name in alternate_paths:
                file_path = alternate_paths[dataset_name]
                logger.info(f"Trying alternate path for {dataset_name}: {file_path}")
            
            if file_path.exists():
                try:
                    if dataset_name == "crop_production":
                        self.data_store.load_crop_production(file_path)
                    elif dataset_name == "crop_calendar":
                        self.data_store.load_crop_calendar(file_path)
                    elif dataset_name == "call_query":
                        self.data_store.load_call_query(file_path)
                    elif dataset_name == "pesticide_recommendations":
                        self.data_store.load_pesticide_recommendations(file_path)
                    elif dataset_name == "pest_solutions":
                        self.data_store.load_pest_solutions(file_path)
                except Exception as e:
                    logger.exception(f"Failed to load dataset {dataset_name}: {e}")
            else:
                logger.warning(f"Dataset file not found: {file_path}")

    def get_model(self, model_name: str):
        return self.models.get(model_name)

    def get_label_map(self, map_name: str) -> Dict[int, str]:
        return self.label_maps.get(map_name, {})

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

# ------------------- Intent helpers -------------------
_intent_proto_cache = {"embeddings": None, "labels": None}

def prepare_intent_prototypes(rag_pipeline: RAGPipeline):
    if _intent_proto_cache["embeddings"] is not None or not rag_pipeline.loaded:
        return
    intents = {
        "faq": ["what is the best fertilizer for rice", "how to plant maize"],
        "fertilizer": ["which fertilizer to use", "recommend fertilizer for my crop"],
        "yield": ["predict my yield", "expected yield for this season"],
        "crop_production": ["rice production in punjab", "wheat yield statistics"],
        "crop_calendar": ["when to sow paddy", "rice harvesting season"],
        "pesticide": ["which pesticide for aphids", "pesticide recommendation"],
        "pest_management": ["how to control stem borer", "pest treatment solution"],
        "agri_qa": ["how do i treat leaf curl", "farming advice"],
        "image_insect": ["identify this insect", "bugs in my crop"],
        "unknown": ["i don't know", "not sure"]
    }
    labels = list(intents.keys())
    embeddings = []
    try:
        if rag_pipeline.embedder:
            for label in labels:
                proto_text = intents[label][0]
                emb = rag_pipeline.embedder.encode(proto_text, convert_to_tensor=False)
                embeddings.append(np.array(emb, dtype='float32'))
            _intent_proto_cache["embeddings"] = np.vstack(embeddings)
            _intent_proto_cache["labels"] = labels
            logger.info("Intent prototypes generated using RAG embedder.")
    except Exception as e:
        logger.exception("Error generating intent prototypes: %s", e)

def classify_intent(text_en: str, rag_pipeline: RAGPipeline) -> Tuple[str, float]:
    """Enhanced intent classification"""
    text_lower = text_en.lower()
    
    # Keyword-based classification with priority order
    # Check for crop production FIRST (before yield)
    if any(word in text_lower for word in ["production data", "production statistics", "crop production", "area production"]):
        return "crop_production", 0.95
    
    # Sowing/Calendar queries
    if any(word in text_lower for word in ["sow", "sowing", "plant", "planting", "harvesting time", "harvest time", "calendar", "season for"]):
        return "crop_calendar", 0.95
    
    # Pesticide queries
    if any(word in text_lower for word in ["pesticide", "insecticide", "spray", "chemical for"]):
        return "pesticide", 0.95
    
    # Pest management (disease/pest control)
    if any(word in text_lower for word in ["control", "treatment", "manage", "disease", "infection", "affected by"]):
        return "pest_management", 0.93
    
    # Fertilizer queries
    if any(word in text_lower for word in ["fertilizer", "npk", "urea", "manure", "fertiliser"]):
        return "fertilizer", 0.95
    
    # Yield prediction (needs features)
    if any(word in text_lower for word in ["predict yield", "yield prediction", "expected yield", "estimate yield"]):
        return "yield", 0.95
    
    # Image classification
    if any(word in text_lower for word in ["insect", "bug", "identify", "image", "photo", "picture"]):
        return "image_insect", 0.99
    
    # Semantic classification using embedder
    if rag_pipeline and rag_pipeline.loaded and _intent_proto_cache["embeddings"] is not None:
        try:
            query_emb = rag_pipeline.embedder.encode(text_en, convert_to_tensor=False)
            query_emb = np.array(query_emb, dtype='float32')
            norm = np.linalg.norm(query_emb) + 1e-12
            query_emb = query_emb / norm
            
            similarities = np.dot(query_emb, _intent_proto_cache["embeddings"].T)
            best_idx = int(np.argmax(similarities))
            best_int = _intent_proto_cache["labels"][best_idx]
            best_conf = float(similarities[best_idx])
            if best_conf > 0.6:
                return best_int, best_conf
        except Exception as e:
            logger.exception("Semantic intent classification error: %s", e)
    
    return "faq", 0.5

# ------------------- Model Routing -------------------
def get_model_response(text_en: str, intent_label: str, context: Dict[str, Any]):
    """Route queries to appropriate models/datasets"""
    rag_pipeline = model_loader.get_model("rag_pipeline")
    data_store = model_loader.data_store
    
    # Handle crop production queries
    if intent_label == "crop_production":
        state = context.get('location') or extract_state_from_query(text_en)
        crop = context.get('crop') or extract_crop_from_query(text_en)
        season = context.get('season') or extract_season_from_query(text_en)
        
        results = data_store.search_crop_production(state, crop, season)
        if results:
            answer = format_crop_production_response(results, state, crop, season)
            return answer, 0.9, "crop_production_dataset"
        else:
            return "I couldn't find crop production data matching your query. Please specify state, crop, or season.", 0.3, "crop_production_dataset"
    
    # Handle crop calendar queries
    if intent_label == "crop_calendar":
        state = context.get('location') or extract_state_from_query(text_en)
        crop = context.get('crop') or extract_crop_from_query(text_en)
        
        results = data_store.search_crop_calendar(state, crop)
        if results:
            answer = format_crop_calendar_response(results, state, crop)
            return answer, 0.9, "crop_calendar_dataset"
        else:
            return "I couldn't find crop calendar information for your query. Please specify state or crop.", 0.3, "crop_calendar_dataset"
    
    # Handle pesticide recommendations
    if intent_label == "pesticide":
        pest_name = extract_pest_from_query(text_en)
        if pest_name:
            pesticides = data_store.search_pesticide_by_pest(pest_name)
            if pesticides:
                answer = f"For **{pest_name}**, the most commonly used pesticides are: **{', '.join(pesticides)}**"
                return answer, 0.95, "pesticide_recommendations"
        return "Please specify the pest name to get pesticide recommendations.", 0.3, "pesticide_recommendations"
    
    # Handle pest management/solutions
    if intent_label == "pest_management":
        pest_name = extract_pest_from_query(text_en)
        crop = context.get('crop') or extract_crop_from_query(text_en)
        
        results = data_store.search_pest_solutions(pest_name, crop)
        if results:
            answer = format_pest_solution_response(results)
            return answer, 0.92, "pest_solutions_dataset"
        else:
            return "I couldn't find pest management solutions for your query. Please provide pest name or crop.", 0.3, "pest_solutions_dataset"
    
    # Handle fertilizer recommendations
    if intent_label == "fertilizer":
        model = model_loader.get_model("fertilizer_recommendation_model")
        label_map = model_loader.get_label_map("fertilizer_label_map")
        
        if not label_map:
            return "Fertilizer model failed: Label map (JSON) missing or empty. Please create the required artifact file.", 0.0, "label_map_error"
        
        if model and context.get('features'):
            ans, conf = predict_fertilizer(context, model, label_map)
            return ans, conf, "fertilizer_recommendation_model"
        else:
            return "To get a fertilizer recommendation, please provide your soil and environmental features (N, P, K values) in the Prediction tab.", 0.4, "needs_context"
    
    # Handle yield predictions
    if intent_label == "yield":
        model = model_loader.get_model("crop_yield_model")
        encoder = model_loader.get_model("crop_yield_encoder")
        if model and encoder and context.get('features'):
            ans, conf = predict_yield(context, model, encoder)
            return ans, conf, "crop_yield_model"
        else:
            return "To get a yield prediction, please provide your crop, field, and environmental features in the Prediction tab.", 0.4, "needs_context"
    
    # Try call_query dataset first (agricultural Q&A)
    if rag_pipeline and rag_pipeline.loaded:
        call_query_answer, call_query_conf = rag_pipeline.search_call_query(text_en)
        if call_query_answer and call_query_conf > 0.5:
            return call_query_answer, call_query_conf, "call_query_dataset"
    
    # Fall back to RAG/FAQ
    if not rag_pipeline or not getattr(rag_pipeline, "loaded", False):
        logger.warning("RAG pipeline requested but not ready.")
        return "The knowledge base (FAQ/RAG) is not fully loaded. Cannot provide an intelligent answer.", 0.0, "rag_fallback"
    
    rag_context = rag_pipeline.search(text_en, top_k=3)
    answer_en, confidence = rag_pipeline.generate_response(text_en, rag_context)
    return answer_en, confidence, "rag_pipeline"

# ------------------- Helper functions for data extraction -------------------
def extract_state_from_query(query: str) -> Optional[str]:
    """Extract state name from query"""
    states = [
        "andhra pradesh", "assam", "bihar", "chhattisgarh", "goa", "gujarat",
        "haryana", "himachal pradesh", "jharkhand", "karnataka", "kerala",
        "madhya pradesh", "maharashtra", "manipur", "meghalaya", "mizoram",
        "nagaland", "odisha", "punjab", "rajasthan", "sikkim", "tamil nadu",
        "telangana", "tripura", "uttar pradesh", "uttarakhand", "west bengal",
        "andaman and nicobar islands"
    ]
    query_lower = query.lower()
    for state in states:
        if state in query_lower:
            return state
    return None

def extract_crop_from_query(query: str) -> Optional[str]:
    """Extract crop name from query"""
    crops = [
        "rice", "wheat", "maize", "bajra", "jowar", "ragi", "cotton", "sugarcane",
        "groundnut", "soybean", "sunflower", "rapeseed", "mustard", "coconut",
        "arecanut", "tea", "coffee", "rubber", "jute", "banana", "mango", "apple",
        "potato", "onion", "tomato", "paddy", "pulses", "cashewnut"
    ]
    query_lower = query.lower()
    for crop in crops:
        if crop in query_lower:
            return crop
    return None

def extract_season_from_query(query: str) -> Optional[str]:
    """Extract season from query"""
    seasons = ["kharif", "rabi", "summer", "whole year", "winter"]
    query_lower = query.lower()
    for season in seasons:
        if season in query_lower:
            return season
    return None

def extract_pest_from_query(query: str) -> Optional[str]:
    """Extract pest name from query"""
    query_lower = query.lower()
    
    # Common pests with variations
    pest_patterns = {
        "stem borer": ["stem borer", "stemborer", "yellow stem borer"],
        "aphids": ["aphid", "aphids", "aphis"],
        "whitefly": ["whitefly", "white fly", "whiteflies"],
        "leaf folder": ["leaf folder", "leaffolder", "leaf roller"],
        "brown plant hopper": ["brown plant hopper", "bph", "plant hopper", "planthopper"],
        "armyworm": ["armyworm", "army worm", "fall armyworm"],
        "bollworm": ["bollworm", "boll worm", "cotton bollworm"],
        "fruit borer": ["fruit borer", "fruit fly"],
        "thrips": ["thrips", "thrip"],
        "jassids": ["jassids", "jassid", "leafhopper", "leaf hopper"],
        "mites": ["mites", "mite", "red mite", "spider mite"],
        "cutworm": ["cutworm", "cut worm"],
        "shoot borer": ["shoot borer", "shootborer"],
        "pod borer": ["pod borer", "podborer"],
        "leaf miner": ["leaf miner", "leafminer"],
        "scale insects": ["scale insect", "scale"],
        "blast": ["blast", "blast disease"],
        "blight": ["blight", "leaf blight", "bacterial blight"],
        "root rot": ["root rot", "rootrot"],
        "wilt": ["wilt", "fusarium wilt"]
    }
    
    # Check for pest patterns
    for canonical_name, variations in pest_patterns.items():
        for variation in variations:
            if variation in query_lower:
                return canonical_name
    
    # Extract from common phrase patterns
    patterns = [
        r"control (\w+(?:\s+\w+)?)",
        r"manage (\w+(?:\s+\w+)?)",
        r"treat (\w+(?:\s+\w+)?)",
        r"(?:pest|insect|disease)\s+(?:is|called)?\s*(\w+(?:\s+\w+)?)",
    ]
    
    import re
    for pattern in patterns:
        match = re.search(pattern, query_lower)
        if match:
            potential_pest = match.group(1).strip()
            # Avoid common words
            if potential_pest not in ["the", "my", "this", "that", "crop", "plant", "field"]:
                return potential_pest
    
    return None

# ------------------- Response formatting functions -------------------
def format_crop_production_response(results: List[Dict], state: str, crop: str, season: str) -> str:
    """Format crop production data into a readable response"""
    if not results:
        return "No production data found."
    
    response = "**Crop Production Data:**\n\n"
    
    for i, record in enumerate(results[:5], 1):
        state_name = record.get('State_Name', 'N/A')
        district = record.get('District_Name', 'N/A')
        crop_name = record.get('Crop', 'N/A')
        year = record.get('Crop_Year', 'N/A')
        season_name = record.get('Season', 'N/A')
        area = record.get('Area', 'N/A')
        production = record.get('Production', 'N/A')
        
        response += f"{i}. **{crop_name}** in **{district}, {state_name}**\n"
        response += f"   - Year: {year} | Season: {season_name}\n"
        response += f"   - Area: {area} hectares | Production: {production} tonnes\n\n"
    
    if len(results) > 5:
        response += f"_Showing 5 of {len(results)} results_"
    
    return response

def format_crop_calendar_response(results: List[Dict], state: str, crop: str) -> str:
    """Format crop calendar data into a readable response"""
    if not results:
        return "No crop calendar information found."
    
    response = "**Crop Calendar Information:**\n\n"
    
    for record in results[:3]:
        state_name = record.get('States/Uts', 'N/A')
        period = record.get('Period', 'N/A')
        
        response += f"**{state_name}** - {period}:\n"
        
        # Show sowing and harvesting info for different crops
        for key, value in record.items():
            if key not in ['States/Uts', 'Period'] and value and str(value) != 'nan':
                response += f"   - {key}: {value}\n"
        response += "\n"
    
    return response

def format_pest_solution_response(results: List[Dict]) -> str:
    """Format pest solution data into a readable response"""
    if not results:
        return "No pest management solutions found."
    
    response = "**Pest Management Solutions:**\n\n"
    
    for i, record in enumerate(results[:3], 1):
        crop = record.get('crop', 'N/A')
        pest = record.get('disease_or_pest', 'N/A')
        agent = record.get('agent', 'N/A')
        treatment = record.get('recommended_treatment', 'N/A')
        treatment_type = record.get('treatment_type', 'N/A')
        active_ingredient = record.get('active_ingredient', 'N/A')
        dosage = record.get('dosage', 'N/A')
        
        response += f"{i}. **{pest}** in **{crop}**\n"
        response += f"   - Agent: {agent}\n"
        response += f"   - Treatment: {treatment} ({treatment_type})\n"
        if active_ingredient and str(active_ingredient) != 'nan':
            response += f"   - Active Ingredient: {active_ingredient}\n"
        if dosage and str(dosage) != 'nan':
            response += f"   - Dosage: {dosage}\n"
        response += "\n"
    
    return response

# ------------------- Prediction helpers -------------------
def create_mock_feature_vector(features: Dict[str, Any]) -> np.ndarray:
    N = features.get('N', 0.0)
    P = features.get('P', 0.0)
    K = features.get('K', 0.0)
    Temp = features.get('Temperature', 0.0)
    Hum = features.get('Humidity', 0.0)
    Mois = features.get('Moisture', 0.0)

    numeric_vector = np.array([Temp, Hum, Mois, N, P, K], dtype='float32')
    ohe_vector = np.zeros(13, dtype='float32')

    if numeric_vector.shape[0] + ohe_vector.shape[0] < 20:
        padding = np.zeros(20 - (numeric_vector.shape[0] + ohe_vector.shape[0]), dtype='float32')
    else:
        padding = np.array([], dtype='float32')

    final_vector = np.concatenate([numeric_vector, ohe_vector, padding])

    if final_vector.shape[0] != 20:
        logger.warning("Feature vector shape mismatch: Expected 20, got %d.", final_vector.shape[0])
        return np.zeros((1, 20), dtype='float32')

    return final_vector.reshape(1, -1).astype('float32')

def predict_yield(context: Dict[str, Any], model_bundle: Any, encoder: Any) -> Tuple[str, float]:
    try:
        if model_bundle is None:
            raise ValueError("Yield model not loaded.")
        features = context.get('features', {})
        mock_features = create_mock_feature_vector(features)
        prediction = model_bundle.predict(mock_features)[0]
        crop_type = features.get('Crop', 'Unknown Crop')
        return f"The predicted crop yield for {crop_type} is approximately **{prediction:,.2f} kg/hectare**.", 0.85
    except Exception as e:
        logger.exception("Yield prediction error: %s", e)
        return "Sorry, I could not process the yield prediction due to an internal model error.", 0.2

def predict_fertilizer(context: Dict[str, Any], model_bundle: Any, label_map: Dict[int, str]) -> Tuple[str, float]:
    try:
        if model_bundle is None:
            raise ValueError("Fertilizer model not loaded.")
        features = context.get('features', {})

        required_keys = ['N', 'P', 'K']
        if not all(k in features for k in required_keys):
            return "Please provide soil parameters (N, P, K) in the Prediction tab for fertilizer recommendation.", 0.3

        mock_features = create_mock_feature_vector(features)
        predicted_label_index = int(model_bundle.predict(mock_features)[0])

        fertilizer_name = label_map.get(predicted_label_index)
        if fertilizer_name is None:
            raise KeyError(f"Predicted index {predicted_label_index} not found in loaded fertilizer label map keys.")

        return f"Based on your soil and crop features, the recommended fertilizer is **{fertilizer_name}**.", 0.95
    except Exception as e:
        logger.exception("Fertilizer prediction error: %s", e)
        return "Sorry, the fertilizer recommendation model encountered an internal error. Please check your input features and model integrity.", 0.2

# ------------------- Remaining functions -------------------
async def create_escalation(query_id: str, query: str, confidence: float, context: Optional[Dict[str, Any]]):
    if db is not None:
        escalation_id = str(uuid.uuid4())
        doc = {
            "escalation_id": escalation_id,
            "query_id": query_id,
            "query": query,
            "confidence": confidence,
            "context": context,
            "timestamp": datetime.utcnow().isoformat()
        }
        logger.info("Escalation created (mock): %s", escalation_id)
        return escalation_id
    return None

def preprocess_image(image: Image.Image, target_size: tuple = (224, 224)) -> np.ndarray:
    image = image.convert("RGB").resize(target_size)
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)

def predict_image(model_name: str, image: Image.Image) -> Tuple[Optional[str], float]:
    model = model_loader.get_model(model_name)
    label_map_name = model_name.replace("classifier", "label_map")
    label_map = model_loader.get_label_map(label_map_name)
    if model is None or not label_map:
        return None, 0.0
    try:
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image, verbose=0)[0]
        predicted_index = int(np.argmax(predictions))
        confidence = float(predictions[predicted_index])
        predicted_label = label_map.get(predicted_index, f"Unknown (Index {predicted_index})")
        return predicted_label, confidence
    except Exception as e:
        logger.exception("Image prediction failed for %s: %s", model_name, e)
        return "Prediction Error", 0.0

# ==================== LIFESPAN ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_client, db
    try:
        db_client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URL, serverSelectionTimeoutMS=5000)
        await db_client.admin.command('ping')
        db = db_client[DB_NAME]
        logger.info("Connected to MongoDB successfully")
    except Exception as e:
        logger.error("MongoDB connection failed: %s. Running without database.", e)
        db = None
    model_loader.load_all_models()
    translator.load_translation_models()
    rag_pipeline = model_loader.get_model("rag_pipeline")
    if rag_pipeline and rag_pipeline.loaded:
        prepare_intent_prototypes(rag_pipeline)
    yield
    if db_client:
        try:
            db_client.close()
            logger.info("Closed MongoDB connection")
        except Exception:
            logger.debug("DB client close failed or not required in environment.")

# ==================== FASTAPI APP ====================
app = FastAPI(
    title="Farmer Advisory System API",
    description="AI-powered agricultural advisory system with multilingual support and comprehensive datasets",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Rate limiting
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host if request.client else "unknown"
    if request.url.path == "/health" or request.url.path.startswith("/debug"):
        return await call_next(request)
    if not await rate_limiter.is_allowed(client_ip):
        return JSONResponse(status_code=429, content={"detail": "Too many requests. Please try again later."})
    return await call_next(request)

# ==================== PYDANTIC MODELS ====================
class ContextData(BaseModel):
    farmer_id: Optional[str] = ""
    location: Optional[str] = ""
    crop: Optional[str] = ""
    season: Optional[str] = ""
    features: Optional[Dict[str, Any]] = None

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    context: Optional[Dict[str, Any]] = None
    request_escalation: bool = False

    @validator("query")
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

class ImageResponse(BaseModel):
    query_id: str
    label: str
    confidence: float
    used_model: str
    timestamp: str
    escalation_id: Optional[str] = None

# ==================== ENDPOINTS ====================
@app.get("/")
async def root():
    return {
        "message": "Farmer Advisory System API",
        "version": "2.0.0",
        "endpoints": {
            "health": "/health",
            "chat": "/api/chat",
            "image": "/api/image",
            "debug": "/debug/model_health"
        }
    }

@app.get("/debug/datasets")
async def debug_datasets():
    """Debug endpoint to check dataset loading"""
    data_store = model_loader.data_store
    
    return {
        "crop_production": {
            "total_records": len(data_store.crop_production_data),
            "sample": data_store.crop_production_data[:2] if data_store.crop_production_data else [],
            "columns": list(data_store.crop_production_data[0].keys()) if data_store.crop_production_data else []
        },
        "crop_calendar": {
            "total_records": len(data_store.crop_calendar_data),
            "sample": data_store.crop_calendar_data[:2] if data_store.crop_calendar_data else [],
            "columns": list(data_store.crop_calendar_data[0].keys()) if data_store.crop_calendar_data else []
        },
        "call_query": {
            "total_records": len(data_store.call_query_data),
            "sample": data_store.call_query_data[:2] if data_store.call_query_data else []
        },
        "pesticide_recommendations": {
            "total_pests": len(data_store.pesticide_recommendations),
            "sample_pests": list(data_store.pesticide_recommendations.keys())[:5]
        },
        "pest_solutions": {
            "total_records": len(data_store.pest_solutions),
            "sample": data_store.pest_solutions[:2] if data_store.pest_solutions else []
        }
    }

@app.get("/debug/model_health")
async def debug_model_health():
    try:
        rp = model_loader.get_model("rag_pipeline")
        return {
            "ok": True,
            "models_loaded": list(model_loader.models.keys()),
            "label_maps": list(model_loader.label_maps.keys()),
            "rag_loaded": bool(rp and rp.loaded),
            "rag_info": {"faq_records": len(rp.faq_data) if rp and rp.faq_data is not None else 0},
            "datasets_loaded": {
                "crop_production": len(model_loader.data_store.crop_production_data),
                "crop_calendar": len(model_loader.data_store.crop_calendar_data),
                "call_query": len(model_loader.data_store.call_query_data),
                "pesticide_recommendations": len(model_loader.data_store.pesticide_recommendations),
                "pest_solutions": len(model_loader.data_store.pest_solutions)
            }
        }
    except Exception as e:
        logger.exception("debug_model_health error: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error during health check.")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": len(model_loader.models),
        "translator_loaded": translator.loaded,
        "database_connected": db is not None,
        "rag_pipeline_ready": getattr(model_loader.rag_pipeline, "loaded", False),
        "datasets_ready": {
            "crop_production": len(model_loader.data_store.crop_production_data) > 0,
            "crop_calendar": len(model_loader.data_store.crop_calendar_data) > 0,
            "call_query": len(model_loader.data_store.call_query_data) > 0,
            "pesticide_recommendations": len(model_loader.data_store.pesticide_recommendations) > 0,
            "pest_solutions": len(model_loader.data_store.pest_solutions) > 0
        },
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    query_id = str(uuid.uuid4())
    user_text = request.query
    detected_lang = "en"
    
    try:
        detected_lang = detect(user_text)
    except Exception:
        detected_lang = "en"
    
    src_lang = detected_lang if detected_lang in ['en', 'hi', 'ml'] else "en"
    user_text_en = translator.translate(user_text, src_lang, "en")
    
    rag_pipeline = model_loader.get_model("rag_pipeline")
    intent_label, answer_conf = classify_intent(user_text_en, rag_pipeline)
    
    answer_en = ""
    used_model = "unknown"

    if "image" in intent_label:
        answer_en = "It looks like you want an image-based diagnosis. Please use the Image Analysis tab to upload an image."
        used_model = instruction_tag("image_upload")
        answer_conf = 1.0
    else:
        try:
            answer_en, answer_conf, model_used = get_model_response(user_text_en, intent_label, request.context or {})
            used_model = model_used
        except Exception as e:
            logger.exception("Error in get_model_response: %s", e)
            answer_en = "Sorry, I couldn't process your query right now. Please try rephrasing or contact support."
            answer_conf = 0.0
            used_model = "fallback"

    escalation_id = None
    if answer_conf < CONFIDENCE_THRESHOLD or request.request_escalation:
        escalation_id = await create_escalation(query_id, user_text, float(answer_conf), request.context)
    
    answer_local = translator.translate(answer_en, "en", src_lang)
    
    doc = {
        "query_id": query_id, "query": user_text, "answer": answer_local, "confidence": float(answer_conf),
        "used_model": used_model, "intent": intent_label, "detected_language": src_lang,
        "escalation_id": escalation_id, "context": request.context or {}, "timestamp": datetime.utcnow().isoformat()
    }
    
    if db is not None:
        logger.info("Mock DB insert: chat_queries")
    
    return ChatResponse(
        query_id=query_id, answer=answer_local, confidence=float(answer_conf), intent=intent_label,
        used_model=used_model, detected_language=src_lang, escalation_id=escalation_id, timestamp=doc["timestamp"]
    )

@app.post("/api/image", response_model=ImageResponse)
async def classify_image(
    file: UploadFile = File(...), 
    crop: Optional[str] = None, 
    location: Optional[str] = None, 
    season: Optional[str] = None
):
    query_id = str(uuid.uuid4())
    try:
        image_bytes = await file.read()
        if len(image_bytes) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File too large.")
        
        image = Image.open(BytesIO(image_bytes))
        context_data = ContextData(crop=crop, location=location, season=season)
        
        best_label, best_conf = predict_image("insect_classifier", image)
        best_model = "insect_classifier"
        
        if best_label is None or best_model is None:
            raise HTTPException(status_code=500, detail="Unable to classify image. Model may not be loaded.")
        
        escalation_id = None
        if best_conf < CONFIDENCE_THRESHOLD:
            escalation_id = await create_escalation(
                query_id, 
                f"Image classification: {file.filename} -> {best_label}", 
                best_conf, 
                context_data.dict()
            )
        
        doc = {
            "query_id": query_id, "file_name": file.filename, "label": best_label, "confidence": best_conf,
            "used_model": best_model, "escalation_id": escalation_id, "timestamp": datetime.utcnow().isoformat()
        }
        
        if db:
            logger.info("Mock DB insert: image_queries")
        
        return ImageResponse(**doc)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Image classification error: %s", e)
        raise HTTPException(status_code=500, detail="An unexpected error occurred during image processing.")

# ==================== Run server ====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT, log_level=LOG_LEVEL.lower())

