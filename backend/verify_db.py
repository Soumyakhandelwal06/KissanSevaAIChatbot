import asyncio
from database import users_collection
from models import UserInDB

async def list_users():
    print("--- Connecting to MongoDB ---")
    users_cursor = users_collection.find()
    users = await users_cursor.to_list(length=100)
    
    print(f"Found {len(users)} users:")
    for user in users:
        print(f"- Name: {user.get('name')}, Phone: {user.get('phone')}, Location: {user.get('location')}")

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(list_users())
