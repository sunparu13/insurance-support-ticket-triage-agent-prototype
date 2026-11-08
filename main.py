import random
import json
import os
import argparse
import time
from transformers import set_seed
from triage_langgraph import TicketDataset, LocalLLM, TicketTriageAgent, DEFAULT_MODEL_PATH, TicketHistoryRAG



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Triage tickets")
    parser.add_argument("--input_csv", type=str, default="./data/aa_dataset-tickets-multi-lang-5-2-50-version.csv", help="Path to the input CSV file containing tickets")
    parser.add_argument("--limit", type=int, default=10, help="Limit the number of tickets to process")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Local model directory (default: ./models/<model name>)")
    parser.add_argument("--output_json", type=str, default=None, help="Output JSON path (triage_results/ with an automatically generated filename)")
    parser.add_argument("--thinking", action="store_true", help="Enable thinking mode")
    parser.add_argument("--data_shuffle_seed", type=int, default=42, help="Random seed for shuffling tickets")
    parser.add_argument("--rag", action="store_true", help="Retrieve historical tickets before classification")
    parser.add_argument("--history_size", type=int, default=5000, help="Number of historical tickets")
    parser.add_argument("--embedding_model", default="./models/Qwen3-Embedding-0.6B",
                        help="Local embedding folder, e.g. ./models/Qwen3-Embedding-0.6B or ./models/Qwen3-Embedding-4B")
    parser.add_argument("--rag_top_k", type=int, default=3, help="Maximum distinct historical tickets per query")
    parser.add_argument("--generation_seed", type=int, default=None, help="Optional generation seed for paired experiments")
    args = parser.parse_args()
    if args.limit < 1:
            parser.error("--limit must be at least 1")

    text_funcs = TicketDataset(args.input_csv)
    #load tickets from csv and process them
    tickets = text_funcs.load_ticket()
    if args.limit > len(tickets):
        parser.error(f"--limit exceeds the number of tickets ({len(tickets)})")
    random.Random(args.data_shuffle_seed).shuffle(tickets)
    selected = tickets[:args.limit]
    
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S") 

    if args.output_json is None:
        model_name = os.path.basename(os.path.normpath(args.model_path))
        args.output_json = os.path.join(
            "triage_results",
            f"triage_results_{timestamp}_{args.limit}_{model_name}_{'thinking' if args.thinking else 'nothinking'}{'_rag' if args.rag else ''}.json"
        )
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)

    retriever = None
    if args.rag:
        excluded_ids = []
        for ticket in selected:
            excluded_ids.append(ticket['ticket_id'])
        history_rag = TicketHistoryRAG()
        history_tickets = text_funcs.load_ticket(include_history=True)
        random.Random(args.data_shuffle_seed).shuffle(history_tickets)
        documents = history_rag.prepare_history_documents(
            history_tickets, excluded_ids, args.history_size, args.data_shuffle_seed,
        )
        retriever = history_rag.build_retriever(documents, args.embedding_model, top_k=args.rag_top_k)

    #initialize the local LLM and the ticket triage agent
    local_llm = LocalLLM(args.model_path, thinking=args.thinking)
    ticket_agent = TicketTriageAgent(local_llm, retriever=retriever, rag_top_k=args.rag_top_k)

    #process "batch" of tickets and save results to JSON
    results = []
    for ticket in selected:
        if args.generation_seed is not None:
            set_seed(args.generation_seed + ticket['ticket_id'])
        results.append(ticket_agent.triage(ticket))
        print(f"Processed {len(results)} out of {args.limit} tickets")

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved results to: {args.output_json}")


#running example: python main.py --input_csv ./data/aa_dataset-tickets-multi-lang-5-2-50-version.csv --limit 10 --model_path ./models/Qwen3-1.7B --thinking --generation_seed 42
#with RAG: python main.py --input_csv ./data/aa_dataset-tickets-multi-lang-5-2-50-version.csv --limit 10 --model_path ./models/Qwen3-1.7B --rag --embedding_model ./models/Qwen3-Embedding-4B --thinking --generation_seed 42 --history_size 5000
