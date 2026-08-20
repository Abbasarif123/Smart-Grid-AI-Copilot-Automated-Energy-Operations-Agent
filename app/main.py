from langchain_core.messages import HumanMessage
from agent.graph import app  # the compiled graph from the previous step


#investigate past electricity spot sprice with a prompt, streams graph through tool calls and LLM reasoning cycles until the agent drafts an action plane
#pauses graph when approval gate is reached and asks human for a decision after showing the draft
#uses decision from the humman and injets it into the state checkpoints abd resumes workflow

def main():
    # define the thread configuration
    # a thread ID is the persistent cursor, the runtime uses it to know which state to resume
    thread_config = {"configurable": {"thread_id": "incident-001"}}
    
    print("--- NEW INCIDENT DETECTED ---")
    initial_input = {
        "messages": [
            HumanMessage(
                content="Electricity spot prices dropped to negative values recently. "
                        "Please query the historical prices from 2024-03-01 to 2024-03-05 and PROPOSE ACTION."
            )
        ]
    }
    
    # run the Graph (phase 1: investigation)
    # the graph will run until it hits the interrupt_before breakpoint
    print("\n[Agent] Investigating logs and drafting action...")
    
    #use stream() to see the output of each node as it finishes
    for event in app.stream(initial_input, config=thread_config): #event is a dict mapping node names to their emitted state updates
        for node_name, state_update in event.items():
            print(f" -> Node '{node_name}' finished execution.")

    #human interception
    # inspect the paused state
    # execution stops here
    current_state = app.get_state(thread_config)
    print("\n--- GRAPH PAUSED FOR HUMAN APPROVAL ---")
    
    # .next property tells us which node is queued up
    print(f"Next queued node: {current_state.next}") 
    
    # extract the last message the agent wrote before pausing
    last_message = current_state.values["messages"][-1].content
    print(f"\n[Agent Draft]:\n{last_message}")
    
    # human-in-the-Loop interception
    user_input = input("\nDo you approve this action? (y/n): ")
    
    if user_input.lower() == 'y':
        # update_state() writes straight to the saved state, creating a new checkpoint
        app.update_state(thread_config, {"human_approved": True})
        print("\n[Operator] Action Approved.")
    else:
        app.update_state(thread_config, {"human_approved": False})
        print("\n[Operator] Action Rejected.")
        
    # resume the Graph (phase 2: execution)
    # the graph is resumed by passing in None for the input, which continues from the breakpoint
    print("\n[System] Resuming graph...")
    for event in app.stream(None, config=thread_config):
        for node_name, state_update in event.items():
            print(f" -> Node '{node_name}' finished execution.")
            
    # final result
    final_state = app.get_state(thread_config)
    print("\n--- WORKFLOW COMPLETE ---")
    print(final_state.values["messages"][-1].content)

if __name__ == "__main__":
    main()