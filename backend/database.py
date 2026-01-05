import motor.motor_asyncio
import os
from dotenv import load_dotenv

load_dotenv()

import certifi

# Use environment variable or default to local
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
DB_NAME = "kissan_seva_db"

client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URL, tlsCAFile=certifi.where())
database = client[DB_NAME]

# Collections
users_collection = database.get_collection("users")
posts_collection = database.get_collection("posts")
