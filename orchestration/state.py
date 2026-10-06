from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class AgentMessage(BaseModel):
    agent: str
    content: str
    data: Optional[Dict[str, Any]] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

class ResearchState(BaseModel):
    user_query: str
    messages: List[AgentMessage] = Field(default_factory=list)
    
    # Structured results from agents
    data_summary: Optional[Dict[str, Any]] = None
    top_designs: Optional[List[Dict[str, Any]]] = None
    ml_insights: Optional[Dict[str, Any]] = None
    physics_explanation: Optional[str] = None
    critic_feedback: Optional[str] = None
    final_report: Optional[str] = None
    
    # Control
    plan: Optional[List[str]] = None
    current_step: int = 0
    is_complete: bool = False

    def add_message(self, agent: str, content: str, data: Dict = None):
        self.messages.append(AgentMessage(agent=agent, content=content, data=data))