from pydantic import BaseModel
from typing import List

class AgentItem(BaseModel):
    url: str
    name: str

class AgentRequest(BaseModel):
    agent: List[AgentItem]