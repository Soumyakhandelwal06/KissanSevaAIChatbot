from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    context: Optional[Dict[str, Any]] = None
    request_escalation: bool = False

class ChatResponse(BaseModel):
    answer: str
    source: str  # "FAQ" or "Agri QA"
