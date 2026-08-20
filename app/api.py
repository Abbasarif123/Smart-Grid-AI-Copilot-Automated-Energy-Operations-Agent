from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
import uuid

# import compiled LangGraph app (rename it to avoid clashing with FastAPI's app)
from agent.graph import app as graph_app 

api = FastAPI(title="Smart Grid Agent API")

#pydantic models for validation
class AlertPayload(BaseModel):
    alert_text: str

class ApprovalPayload(BaseModel):
    approved: bool

#API Endpoints

@api.post("/api/investigate")
def start_investigation(payload: AlertPayload):
    """
    Endpoint 1: Receives an alert, starts the agent, and pauses for approval.
    """
    # generate a unique thread ID for this specific incident
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    initial_input = {
        "messages": [HumanMessage(content=payload.alert_text)]
    }
    
    # run the graph. it should execute until it hits 'interrupt_before'
    # use invoke() instead of stream() because want the final paused state for the API response
    graph_app.invoke(initial_input, config=config)
    
    # retrieve the current paused state
    state = graph_app.get_state(config)
    
    # if it didn't pause (like lets say no action was proposed), return the final result
    if not state.next:
        return {
            "status": "completed",
            "thread_id": thread_id,
            "message": state.values["messages"][-1].content
        }
        
    # if it paused, return the drafted action to the frontend for the user to review
    return {
        "status": "pending_approval",
        "thread_id": thread_id,
        "drafted_action": state.values["messages"][-1].content,
        "next_node": state.next[0]
    }

@api.post("/api/approve/{thread_id}")
def approve_action(thread_id: str, payload: ApprovalPayload):
    """
    Endpoint 2: Receives the human's decision and resumes the paused graph.
    """
    config = {"configurable": {"thread_id": thread_id}}
    
    # verify the thread actually exists and is currently paused
    state = graph_app.get_state(config)
    if not state.next:
        raise HTTPException(status_code=400, detail="Graph is not paused or thread_id is invalid.")
        
    # update the graph's memory with the human's decision
    graph_app.update_state(config, {"human_approved": payload.approved})
    
    # resume execution by passing None as the input
    graph_app.invoke(None, config=config)
    
    # get the final outcome after execution completes
    final_state = graph_app.get_state(config)
    
    return {
        "status": "completed",
        "thread_id": thread_id,
        "final_message": final_state.values["messages"][-1].content
    }