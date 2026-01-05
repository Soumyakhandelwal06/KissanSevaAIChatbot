from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class UserBase(BaseModel):
    phone: str = Field(..., min_length=10, max_length=15)
    name: str = Field(..., min_length=1)
    location: str
    crop: str
    land_size: str

class UserCreate(UserBase):
    password: str = Field(..., min_length=6)

class UserLogin(BaseModel):
    phone: str
    password: str

class UserInDB(UserBase):
    hashed_password: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

class PostBase(BaseModel):
    content: str
    image_url: Optional[str] = None
    location: Optional[str] = None

class PostCreate(PostBase):
    pass

class PostInDB(PostBase):
    id: str # MongoDB ObjectId as string
    user_name: str
    likes: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
