import os
import json
import asyncio
import logging
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from google.genai.errors import APIError

# ===================== LOGGING =====================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===================== ENV =====================
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set")

API_HOST = "0.0.0.0"
API_PORT = 8000

# ===================== GEMINI CLIENT =====================
client = genai.Client(api_key=GEMINI_API_KEY)

# Use gemini-flash-latest for better free tier stability
MODEL_NAME = "gemini-flash-latest"

# ===================== FASTAPI =====================
app = FastAPI(title="KissanSeva AI (Gemini)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== MODELS =====================
class ChatContext(BaseModel):
    crop: Optional[str] = "general"
    location: Optional[str] = "India"
    season: Optional[str] = "all"

class ChatRequest(BaseModel):
    query: str
    context: Optional[ChatContext] = ChatContext()

class ChatResponse(BaseModel):
    answer: str
    confidence: float
    intent: str
    used_model: str

class ImageResponse(BaseModel):
    label: str
    confidence: float
    remedy: str
    used_model: str

# ===================== HELPERS =====================
async def generate_content_with_retry(contents, config):
    """Helper to handle Gemini API rate limits with retries."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Using sync call for now as per original code structure
            return client.models.generate_content(
                model=MODEL_NAME,
                contents=contents,
                config=config,
            )
        except Exception as e:
            err_msg = str(e).lower()
            if ("429" in err_msg or "resource_exhausted" in err_msg) and attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5
                logger.warning(f"Gemini Rate Limit hit. Retrying in {wait_time}s... (Attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(wait_time)
                continue
            raise e

def build_system_prompt(ctx: ChatContext) -> str:
    return (
        "You are an expert Krishi Officer (Agricultural Advisor) for India. "
        "Give practical, accurate farming advice.\n"
        "If the user writes in Malayalam, reply fully in Malayalam.\n"
        "If the user writes in Hindi, reply fully in Hindi.\n"
        "Use bullet points and simple language.\n\n"
        f"Context:\nCrop: {ctx.crop}\nLocation: {ctx.location}\nSeason: {ctx.season}"
    )

def detect_intent(query: str) -> str:
    q = query.lower()
    if any(x in q for x in ["disease", "spot", "leaf", "fungus"]):
        return "CROP_DISEASE"
    if any(x in q for x in ["pest", "insect", "bug"]):
        return "PEST_CONTROL"
    if any(x in q for x in ["fertilizer", "npk"]):
        return "FERTILIZER"
    if any(x in q for x in ["price", "market", "mandi"]):
        return "MARKET_PRICE"
    return "GENERAL_ADVICE"

# ===================== ROUTES =====================
@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        system_prompt = build_system_prompt(request.context)
        intent = detect_intent(request.query)

        resp = await generate_content_with_retry(
            contents=request.query,
            config=genai.types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.3,
            ),
        )

        return ChatResponse(
            answer=resp.text.strip(),
            confidence=0.9 if intent != "MARKET_PRICE" else 0.65,
            intent=intent,
            used_model=MODEL_NAME,
        )

    except APIError as e:
        raise HTTPException(status_code=500, detail=f"Gemini API Error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/image", response_model=ImageResponse)
async def analyze_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        if len(image_bytes) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Image too large (max 10MB)")

        image_part = genai.types.Part.from_bytes(
            data=image_bytes,
            mime_type=file.content_type,
        )

        system_prompt = (
            "You are a crop disease and pest expert.\n"
            "Analyze the image and return STRICT JSON with keys:\n"
            "label, confidence, remedy\n"
            "confidence must be between 0.0 and 1.0"
        )

        resp = await generate_content_with_retry(
            contents=[image_part, "Analyze this crop image"],
            config=genai.types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
            ),
        )

        data = json.loads(resp.text)

        return ImageResponse(
            label=data["label"],
            confidence=float(data["confidence"]),
            remedy=data["remedy"],
            used_model=MODEL_NAME,
        )

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid JSON from Gemini")
    except APIError as e:
        raise HTTPException(status_code=500, detail=f"Gemini API Error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ===================== RUN =====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT, reload=True)
