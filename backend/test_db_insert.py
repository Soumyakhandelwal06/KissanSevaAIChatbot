import asyncio
from database import users_collection
from datetime import datetime
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

async def create_test_user():
    print("--- Attempting to create test user ---")
    
    test_phone = "9462783629" # User's phone from logs
    test_pass = "Soumya@123"  # User's pass from logs
    
    # Check if exists
    existing = await users_collection.find_one({"phone": test_phone})
    if existing:
        print(f"User {test_phone} already exists. Deleting...")
        await users_collection.delete_one({"phone": test_phone})
    
    user_doc = {
        "name": "Test User",
        "phone": test_phone,
        "location": "Test Location",
        "crop": "Rice",
        "land_size": "2 Acres",
        "hashed_password": pwd_context.hash(test_pass),
        "created_at": datetime.utcnow()
    }
    
    result = await users_collection.insert_one(user_doc)
    print(f"User inserted with ID: {result.inserted_id}")

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(create_test_user())
