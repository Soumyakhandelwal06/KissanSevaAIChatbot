
import os
from typing import Optional, Dict, Any, Union
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai.errors import APIError
import dotenv
import json

# --- CONFIGURATION (Assuming these are defined elsewhere) ---
# from temp import API_HOST, API_PORT, LOG_LEVEL
# NOTE: Using environment variables for host/port/log level is best practice, 
# but for a simple test, you can set them here if 'temp' file is an issue:
API_HOST = "0.0.0.0"
API_PORT = 8000
LOG_LEVEL = "info"

dotenv.load_dotenv()


# --- 1. CONFIGURATION ---
# IMPORTANT: The environment variable GEMINI_API_KEY must be set
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    # Use print instead of raise for better server startup control if running in a container
    print("WARNING: GEMINI_API_KEY environment variable not set. Gemini endpoints will fail.")

# Initialize the Gemini Client
client = None
if GEMINI_API_KEY:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"Error initializing Gemini client: {e}")


# --- 2. FASTAPI SETUP ---
app = FastAPI(title="KissanSeva AI Backend")

# Configure CORS to allow your React frontend (running on default 5173 for Vite)
origins = [
    "http://localhost:5173",    # <-- VITE Frontend URL (REQUIRED)
    "http://127.0.0.1:5173",    # <-- VITE Frontend IP (Good practice)
    "http://localhost:3000",    # (Optional: Previous default)
    "http://127.0.0.1:3000",    # (Optional)
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. Pydantic Models for Data Validation ---

class Context(BaseModel):
    """Farming context data."""
    crop: str = "general"
    location: str = "India"
    season: str = "all"
    # Removed: features: Optional[Dict[str, Union[float, str]]] = None 

class ChatRequest(BaseModel):
    """Model for text-based farmer query."""
    query: str
    context: Context
    # Removed: request_escalation: bool = False (can be kept if needed for manual flags)

class ChatResponse(BaseModel):
    """Model for the AI's text response (Simplified for Gemini's direct output)."""
    answer: str
    # Keeping these simulated fields for UI consistency
    intent: str
    confidence: float
    used_model: str
    escalation_id: Optional[str] = None

class ImageResponse(BaseModel):
    """Model for the AI's image analysis response."""
    label: str
    confidence: float
    remedy: str
    used_model: str
    escalation_id: Optional[str] = None


# --- 4. CORE LOGIC FUNCTIONS ---

# Removed: _simulate_prediction_model

def _generate_contextual_prompt(query: str, context: Context) -> tuple[str, str]:
    """Creates a detailed system prompt for Gemini to act as an expert Krishi Officer."""
    
    # Standard chat prompt for Gemini
    system_prompt = (
        "You are an expert Krishi Officer (Agricultural Advisor) for Kerala, India, specializing in local crops and schemes. "
        "Your advice must be accurate, practical, and highly specific to the context provided. "
        "Respond in the language of the query. If the query is in **Malayalam**, respond entirely in **Malayalam**. "
        "Structure your answer with clear headings and bullet points. "
        f"**FARMING CONTEXT:** Crop: {context.crop}, Location: {context.location}, Season: {context.season}."
    )
    user_query = query
    
    return system_prompt, user_query

# --- 5. API ENDPOINTS ---

@app.get("/health")
def health_check():
    """Simple endpoint for the frontend to check if the server is running."""
    return {"status": "ok", "service": "KissanSeva AI Backend"}

@app.post("/api/chat", response_model=ChatResponse)
async def chat_query(request: ChatRequest):
    """Handles text-based queries using the Gemini API."""
    if not client:
        raise HTTPException(status_code=503, detail="Gemini client not initialized. Check API Key.")
    
    # Use the simplified Gemini chat logic
    system_prompt, user_query = _generate_contextual_prompt(request.query, request.context)

    try:
        # 1. Main Gemini Call for Answer
        response = client.models.generate_content(
            model='gemini-2.5-pro',
            contents=user_query,
            config=genai.types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.3 # Slightly higher for more creative advice if needed
            )
        )
        
        # 2. Simulated Intent Detection (Still using a quick Gemini call)
        intent_response = client.models.generate_content(
            model='gemini-2.5-pro',
            contents=f"Classify the following query into one of these intents: PEST_CONTROL, FERTILIZER_RECOMMENDATION, CROP_ADVICE, MARKET_PRICE, SCHEME_INFO. Query: '{request.query}'",
            config=genai.types.GenerateContentConfig(temperature=0.0)
        )
        # Attempt to clean up the intent string
        intent = intent_response.text.split(":")[0].strip().replace('\n', '')
        
        # 3. Simulated Confidence/Escalation (For UI demonstration)
        confidence = 0.90
        escalation_id = None
        if "market price" in request.query.lower() or "latest scheme" in request.query.lower():
            # Force low confidence for real-time data or data outside LLM's core knowledge
            confidence = 0.65
            escalation_id = f"ESC-{os.urandom(4).hex()}"
        
        return ChatResponse(
            answer=response.text.strip(),
            intent=intent,
            confidence=confidence,
            used_model='KissanSeva AI Model',
            escalation_id=escalation_id
        )

    except APIError as e:
        raise HTTPException(status_code=500, detail=f"Gemini API Error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


@app.post("/api/image", response_model=ImageResponse)
async def analyze_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    crop: str = Form("general"),
    location: str = Form("India"),
    season: str = Form("all")
):
    """Handles image uploads for multi-modal analysis (e.g., disease detection) using Gemini."""
    if not client:
        raise HTTPException(status_code=503, detail="Gemini client not initialized. Check API Key.")
    
    # Check file size (optional, but good practice)
    MAX_FILE_SIZE = 10 * 1024 * 1024 # 10MB
    file_contents = await file.read()
    if len(file_contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large. Max 10MB.")
        
    try:
        
        # Prepare the Gemini `Part` object
        image_part = genai.types.Part.from_bytes(
            data=file_contents,
            mime_type=file.content_type
        )

        # System and User Prompt for Image Analysis
        system_prompt = (
            "You are an expert Krishi Officer specializing in crop pathology. "
            "Analyze the uploaded image and classify the disease/pest, estimate confidence, "
            "and provide a practical remedy. Respond in a strict JSON format with the keys: "
            "'label', 'confidence', 'remedy'. Confidence must be a float between 0.0 and 1.0. "
            f"FARMING CONTEXT: Crop: {crop}, Location: {location}, Season: {season}."
        )

        user_prompt = "Analyze the image and provide diagnosis and remedy."
        
        # Call the multi-modal model, requesting JSON output
        response = client.models.generate_content(
            model='gemini-2.5-pro',
            contents=[image_part, user_prompt],
            config=genai.types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                response_schema={"type": "object", "properties": {
                    "label": {"type": "string", "description": "Disease or pest name."},
                    "confidence": {"type": "number", "description": "Confidence score (0.0 to 1.0)."},
                    "remedy": {"type": "string", "description": "Practical treatment steps."},
                }, "required": ["label", "confidence", "remedy"]},
            )
        )
        
        # Parse the JSON response
        analysis_result = json.loads(response.text)
        
        # --- Simulated Escalation Logic for Image Analysis ---
        escalation_id = None
        if analysis_result.get('confidence', 0.0) < 0.7:
            # Low confidence triggers escalation
            escalation_id = f"ESC-IMG-{os.urandom(4).hex()}"

        return ImageResponse(
            label=analysis_result['label'],
            confidence=analysis_result['confidence'],
            remedy=analysis_result['remedy'],
            used_model='KissanSeva AI Model',
            escalation_id=escalation_id
        )

    except APIError as e:
        raise HTTPException(status_code=500, detail=f"Gemini API Error: {e}")
    except json.JSONDecodeError:
        # This handles cases where the LLM does not return clean JSON
        raise HTTPException(status_code=500, detail="Failed to parse JSON response from LLM. Check system prompt.")
    except Exception as e:
        # Catch all other errors, including file processing
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during image processing: {e}")


# # ==================== Run server ====================
if __name__ == "__main__":
    import uvicorn
    # Make sure to run the server with: uvicorn main:app --reload
    uvicorn.run(app, host=API_HOST, port=API_PORT, log_level=LOG_LEVEL.lower())