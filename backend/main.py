import os
import json
import asyncio
import logging
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from google.genai.errors import APIError
import random
import requests

# ===================== LOGGING =====================
import logging
from datetime import datetime
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== MODELS =====================
class ChatContext(BaseModel):
    crop: Optional[str] = "general"
    location: Optional[str] = "India"
    season: Optional[str] = "all"
    language: Optional[str] = "English"

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
    remedy: str
    confidence: float
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
async def analyze_image(
    file: UploadFile = File(...),
    query: Optional[str] = Form(None),
    language: Optional[str] = Form("English")
):
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
            "confidence must be between 0.0 and 1.0\n"
        )
        
        user_prompt = "Analyze this crop image"
        if query and query.strip():
            user_prompt = f"User Question: {query}\n\nAnalyze the crop image based on the user's question."

        resp = await generate_content_with_retry(
            contents=[image_part, user_prompt],
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

from passlib.context import CryptContext
from database import users_collection, posts_collection
from models import UserCreate, UserLogin, UserInDB, PostCreate, PostInDB

# ... (Auth Helpers verify_password, get_password_hash remain same) ...

# ===================== COMMUNITY ROUTES =====================
@app.get("/api/posts")
async def get_posts():
    posts_cursor = posts_collection.find().sort("created_at", -1).limit(50)
    posts = await posts_cursor.to_list(length=50)
    # Convert ObjectId to str for JSON serialization
    for p in posts:
        p["id"] = str(p["_id"])
        del p["_id"]
    return posts

@app.post("/api/posts")
async def create_post(post: PostCreate, user_name: str, location: str):
    post_dict = post.dict()
    post_dict["user_name"] = user_name
    post_dict["location"] = location
    post_dict["created_at"] = datetime.utcnow()
    post_dict["likes"] = 0
    
    new_post = await posts_collection.insert_one(post_dict)
    
    # Return the created post
    created_post = post_dict
    created_post["id"] = str(new_post.inserted_id)
    return created_post

# ===================== AUTH HELPERS =====================
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

# ===================== AUTH ROUTES =====================
@app.post("/api/register")
async def register(user: UserCreate):
    # Check if phone already exists
    existing_user = await users_collection.find_one({"phone": user.phone})
    if existing_user:
        raise HTTPException(status_code=400, detail="Phone number already registered")

    # Hash password
    hashed_password = get_password_hash(user.password)
    
    # Create user dict
    user_dict = user.dict()
    user_dict["hashed_password"] = hashed_password
    del user_dict["password"]
    user_dict["created_at"] = datetime.utcnow() # Simple timestamp

    # Insert into DB
    result = await users_collection.insert_one(user_dict)
    
    return {
        "status": "success",
        "message": "User registered successfully",
        "user": {
            "name": user.name,
            "phone": user.phone,
            "location": user.location,
            "crop": user.crop,
            "land_size": user.land_size
        }
    }

@app.post("/api/login")
async def login(user_data: UserLogin):
    user = await users_collection.find_one({"phone": user_data.phone})
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect phone number or password")
    
    if not verify_password(user_data.password, user["hashed_password"]):
        raise HTTPException(status_code=400, detail="Incorrect phone number or password")

    return {
        "status": "success",
        "message": "Login successful",
        "user": {
            "name": user["name"],
            "phone": user["phone"],
            "location": user["location"],
            "crop": user["crop"],
            "land_size": user["land_size"]
        }
    }

# ===================== MARKET ROUTES =====================
@app.get("/api/mandi")
async def get_mandi(state: Optional[str] = "Maharashtra", commodity: Optional[str] = "Onion"):
    # Try real API if key exists
    api_key = os.getenv("DATA_GOV_API_KEY")
    data = []
    
    # Simple dictionary for localized mock markets
    mock_markets = {
        "Andhra Pradesh": ["Guntur", "Kurnool", "Vijayawada", "Tirupati"],
        "Arunachal Pradesh": ["Naharlagun", "Pasighat", "Namsai"],
        "Assam": ["Guwahati", "Jorhat", "Silchar", "Tezpur"],
        "Bihar": ["Patna", "Muzaffarpur", "Gaya", "Purnia"],
        "Chhattisgarh": ["Raipur", "Bilaspur", "Durg", "Rajnandgaon"],
        "Delhi": ["Azadpur", "Okhla", "Keshopur", "Ghazipur"],
        "Goa": ["Mapusa", "Margao", "Ponda"],
        "Gujarat": ["Ahmedabad", "Surat", "Rajkot", "Vadodara", "Amreli"],
        "Haryana": ["Karnal", "Ambala", "Hisar", "Rohtak"],
        "Himachal Pradesh": ["Shimla", "Solan", "Kangra"],
        "Jammu and Kashmir": ["Srinagar", "Jammu", "Udhampur"],
        "Jharkhand": ["Ranchi", "Bokaro", "Dhanbad", "Jamshedpur"],
        "Karnataka": ["Bengaluru", "Mysuru", "Hubballi", "Belagavi", "Raichur"],
        "Kerala": ["Kochi", "Thiruvananthapuram", "Kozhikode", "Thrissur"],
        "Madhya Pradesh": ["Indore", "Bhopal", "Gwalior", "Jabalpur", "Ujjain"],
        "Maharashtra": ["Pune", "Nashik", "Nagpur", "Mumbai", "Aurangabad", "Lasalgaon"],
        "Manipur": ["Imphal", "Thoubal", "Bishnupur"],
        "Meghalaya": ["Shillong", "Tura", "Jowai"],
        "Mizoram": ["Aizawl", "Lunglei", "Champhai"],
        "Nagaland": ["Dimapur", "Kohima", "Mokokchung"],
        "Odisha": ["Bhubaneswar", "Cuttack", "Rourkela", "Sambalpur"],
        "Punjab": ["Ludhiana", "Patiala", "Bhatinda", "Jalandhar", "Khanna"],
        "Rajasthan": ["Jaipur", "Jodhpur", "Kota", "Bikaner", "Sri Ganganagar"],
        "Sikkim": ["Gangtok", "Namchi", "Geyzing"],
        "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai", "Salem", "Trichy"],
        "Telangana": ["Hyderabad", "Warangal", "Nizamabad", "Khammam"],
        "Tripura": ["Agartala", "Udaipur", "Dharmanagar"],
        "Uttar Pradesh": ["Lucknow", "Agra", "Kanpur", "Varanasi", "Meerut"],
        "Uttarakhand": ["Dehradun", "Haridwar", "Haldwani", "Rudrapur"],
        "West Bengal": ["Kolkata", "Siliguri", "Burdwan", "Malda", "Medinipur"]
    }

    if api_key:
        try:
             # This is a sample URL for data.gov.in Mandi API
             url = f"https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070?api-key={api_key}&format=json&limit=10&filters[state]={state}&filters[commodity]={commodity}"
             resp = requests.get(url, timeout=5)
             if resp.status_code == 200:
                 real_data = resp.json()
                 records = real_data.get("records", [])
                 for r in records:
                     data.append({
                         "market": r.get("market"),
                         "min_price": int(float(r.get("min_price", 0))),
                         "max_price": int(float(r.get("max_price", 0))),
                         "modal_price": int(float(r.get("modal_price", 0))),
                         "date": r.get("arrival_date")
                     })
        except Exception as e:
            logger.error(f"Failed to fetch mandi data: {e}")

    # Fallback / Demo Data
    if not data:
        markets = mock_markets.get(state, ["Local Mandi 1", "Local Mandi 2", "Local Mandi 3"])
        # Base price variation by commodity type
        base = 2500 # Default
        c = commodity.lower()
        
        # Vegetables
        if c in ["onion", "potato", "tomato", "cauliflower", "cabbage", "brinjal", "okra", "carrot"]: base = 1500
        if c in ["garlic", "ginger", "chilli"]: base = 6000
        if c in ["spinach", "coriander", "fenugreek"]: base = 1000
        
        # Fruits
        if c in ["apple", "banana", "mango", "pomegranate", "lemon", "papaya", "guava", "orange"]: base = 4000
        if c in ["grapes", "pineapple"]: base = 5000
        
        # Grains/Cereals
        if c in ["rice", "wheat", "maize", "barley", "millet", "jowar", "bajra"]: base = 2200
        
        # Pulses/Dal
        if c in ["moong", "urad", "tur", "masoor", "gram", "arhar"]: base = 7000
        
        # Commercial/Spices
        if c in ["cotton", "soybean", "jute"]: base = 5500
        if c in ["turmeric", "cumin", "cardamom", "pepper"]: base = 12000
        if c in ["sugarcane", "mustard", "groundnut"]: base = 4500
        
        for m in markets:
            mp = base + random.randint(-200, 200)
            data.append({
                "market": m,
                "min_price": mp - 100,
                "max_price": mp + 150,
                "modal_price": mp,
                "date": datetime.now().strftime("%d/%m/%Y")
            })
            
    # Generate Trend Data (Last 7 days)
    trend = []
    current_val = float(data[0]['modal_price']) if data else 2000.0
    
    # 7 days trend
    import datetime as dt
    today = datetime.now()
    for i in range(7):
        d = today - dt.timedelta(days=6-i)
        trend.append({
             "date": d.strftime("%d %b"),
             "price": int(current_val + random.randint(-150, 150))
        })

    return {"current": data, "trend": trend}

# ===================== WEATHER ROUTES =====================
@app.get("/api/weather")
async def get_weather(city: str):
    try:
        # Step 1: Geocoding
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
        geo_res = requests.get(geo_url, timeout=5)
        if geo_res.status_code != 200:
            return JSONResponse(status_code=404, content={"message": "City not found"})
            
        geo_data = geo_res.json()
        if not geo_data.get("results"):
            return JSONResponse(status_code=404, content={"message": "City not found"})
            
        location = geo_data["results"][0]
        lat = location["latitude"]
        lon = location["longitude"]
        place_name = f"{location['name']}, {location.get('admin1', '')}, {location.get('country', '')}"

        # Step 2: Forecast
        # Daily: max temp, min temp, rain sum, weathercode
        forecast_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=auto"
        weather_res = requests.get(forecast_url, timeout=5)
        
        if weather_res.status_code != 200:
            return JSONResponse(status_code=500, content={"message": "Weather service unavailable"})
            
        w_data = weather_res.json()
        daily = w_data.get("daily", {})
        
        forecast = []
        if daily:
            times = daily.get("time", [])
            max_temps = daily.get("temperature_2m_max", [])
            min_temps = daily.get("temperature_2m_min", [])
            precips = daily.get("precipitation_sum", [])
            codes = daily.get("weather_code", [])
            
            for i in range(len(times)):
                forecast.append({
                    "date": times[i],
                    "max_temp": max_temps[i],
                    "min_temp": min_temps[i],
                    "rain": precips[i],
                    "code": codes[i]
                })

        return {
            "location": place_name,
            "forecast": forecast
        }

    except Exception as e:
        logger.error(f"Weather API Error: {e}")
        return JSONResponse(status_code=500, content={"message": "Internal Server Error"})

# ===================== RUN =====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=API_HOST, port=API_PORT, reload=True)
