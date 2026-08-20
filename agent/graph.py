from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from agent.state import AgentState
from agent.tools import query_historical_spot_prices


#initialise llm and bind tools to it
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tools = [query_historical_spot_prices]
llm_with_tools = llm.bind_tools(tools)

#define the nodes
def agent_reasoning_node(state: AgentState):
    """
    The main LLM node that decides 
    whether to call a tool or give a final answer
    """
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

tool_node = ToolNode(tools)

def human_approval_node(state: AgentState):
    """
    A passive node
    The graph will be configured to interrupt BEFORE this node
    When resumed, it checks if the human approved the action
    """
    if state.get("human_approved"):
        return {"messages": [{"role": "assistant", "content": "Action approved by operator. Executing..."}]}
    else:
        return {"messages": [{"role": "assistant", "content": "Action rejected by operator. Aborting."}]}


#define routing logic
def should_continue(state: AgentState):
    """Determines if the agent needs to use a tool, ask for approval, or finish"""

    last_message = state["messages"][-1]
    
    #if the LLM decided to call a tool, route to the tool node
    if last_message.tool_calls:
        return "tools"
    
    # if the LLM proposes an action, route to human approval
    if "PROPOSE ACTION:" in last_message.content:
        return "human_approval"
        
    return END    


#compile the graph
workflow = StateGraph(AgentState)

#add nodes
workflow.add_node("agent", agent_reasoning_node)
workflow.add_node("tools", tool_node)
workflow.add_node("human_approval", human_approval_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, ["tools", "human_approval", END])
workflow.add_edge("tools", "agent")
workflow.add_edge("human_approval", END)

# checkpointer is required to save state during human-in-the-loop pauses
memory = InMemorySaver()

#=set the graph to automatically pause execution BEFORE the human_approval node runs
app = workflow.compile(
    checkpointer=memory, 
    interrupt_before=["human_approval"]
)