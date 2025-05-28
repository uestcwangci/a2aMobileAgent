from pydantic import BaseModel
from typing import List

class AgentRequest(BaseModel):
    url: List[str]
    name: List[str]