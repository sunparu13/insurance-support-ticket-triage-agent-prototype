from triage_langgraph import TicketTriageAgent, LocalLLM

LANGGRAPH_PNG_PATH = "triage_graph.png"
MODEL_PATH = "./models/Qwen3-1.7B"    

local_llm = LocalLLM(MODEL_PATH)
ticket_agent = TicketTriageAgent(local_llm)

graph = ticket_agent.build_graph()

graph.get_graph().draw_mermaid_png(
    output_file_path=LANGGRAPH_PNG_PATH
)