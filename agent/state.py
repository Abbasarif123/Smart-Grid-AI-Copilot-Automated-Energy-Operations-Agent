from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from operator import add

class AgentState(TypedDict):
    # 'messages' key must use the 'add' operator so new messages are appended,
    # DONOT overwriting the entire list at each step.
    messages: Annotated[Sequence[BaseMessage], add]
    
    # track custom state variables for our specific energy usecase
    anomaly_detected: bool
    proposed_action: str
    human_approved: bool