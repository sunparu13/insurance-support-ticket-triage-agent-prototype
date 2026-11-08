# Insurance Support Ticket Triage Agent

A local prototype that support tickets (as insurance background), then recommends an action route as well as asks for clarification. It does not execute actions or process follow-up replies.

## Setup

Use Python 3.12. Tested on all MacOS (M1 with 16 GB RAM), Windows 11 (RTX3070 8GB), and Ubuntu 24.04. Run commands from the project root:

```bash
python -m venv .venv # create virtual environment
source .venv/bin/activate #activate virtual env
python -m pip install -r requirements.txt #install required dependencies
python dataset_download.py #download dataset
hf download Qwen/Qwen3-1.7B --local-dir ./models/Qwen3-1.7B #download a model with given folder name
```

On Windows, activate virtual env with `.venv\Scripts\activate` instead. The dataset ([Customer IT Support dataset](https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets)) download needs Kaggle authentication in default into `data/`.

## Project Layout

```text
.
├── README.md                   # Setup and usage instructions
├── Technical_Document.md       # Tech Docu A-F
├── requirements.txt            # Python dependencies
├── main.py                     # CLI agent start
├── triage_langgraph.py          # core functions and classes of agent
├── prompt.py                   # agent text prompts
├── evaluate.py                 # evluation agent with customized metrics and human evaluation method
├── dataset_download.py         # Kaggle download
├── data_overview.ipynb          # Dataset overview
├── langgraph_export.py         # Workflow diagram export
├── triage_graph.png            # Exported workflow diagram
├── tests/
│   ├── test_agent_workflow.py   # Scenario tests based on pytest
│   └── scenario_review_20_cases_Qwen3-1.7B_thinking.csv
├── triage_results/             # Saved agent predictions
├── evaluation_results/         # Evaluation for saved predictions
├── triage_human_review_2026-11-07_18-20-04_300_Qwen3-1.7B_thinking.csv #generated human review within evalluation
├── data/                       # Downloaded datasets (not committed)
└── models/                     # Local generation and embedding models (not committed)
```

## Run Triage Agent
In terminal, run the following command to start the agent (Note: for RAG, download embedding model first):
```bash
python main.py --input_csv ./data/aa_dataset-tickets-multi-lang-5-2-50-version.csv --limit 300 --model_path ./models/Qwen3-1.7B --embedding_model ./models/Qwen3-Embedding-4B --thinking --generation_seed 42 --history_size 5000 --rag --rag_top_k 3
```

The recommended configuration is **Qwen3-1.7B with thinking**, based on performance, running devices and running time. More detail in [Technical Document - Results and Limitations](Technical_Document.md#saved-experiment-results).

Tickets are processed sequentially. The default dataset is `data/aa_dataset-tickets-multi-lang-5-2-50-version.csv`. Omitting `--thinking` disables reasoning mode. `--data_shuffle_seed 42` fixes ticket selection. Use `--generation_seed 42` to also seed model generation. Outputs may still vary across hardware and software environments. Use `--limit` to control how many ticket you want to triage in one script run. Run `python main.py --help` for more detailed explanations of each argument.

Results include `ticket_id`, `text_content`, `topic`, `urgency`, `decision`, `classification_reason`, `next_action`, `status`, and `notes`. When RAG is enabled, results also include `retrieved_ticket_ids`. They are saved under `triage_results/` with a timestamped filename.

### Optional Historical Retrieval
An embedding model is required. Here use [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) (default) or [Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B).

```bash
hf download Qwen/Qwen3-Embedding-0.6B --local-dir ./models/Qwen3-Embedding-0.6B
```
With `python main.py --xxx --rag --rag_top_k 3 --history_size 5000`, the classifier retrieves up to `rag_top_k` (default: 3) historical tickets from a local Chroma index. Normal triage uses `TicketDataset.load_ticket()`, which returns only ID, text, and language. RAG loads historical reference fields separately with `load_ticket(include_history=True)`. 

The default history is 5,000 randomly sampled tickets, after excluding the IDs of all tickets selected for the current run. `RecursiveCharacterTextSplitter` splits long tickets into 1000 character chunks with 100 character overlap. Based on the stats in `data_overview.ipynb`, chunk size is enough to contain an entire ticket in most cases. Priority, queue, type, and tags are historical references, not answers for the current ticket.

```bash
hf download Qwen/Qwen3-Embedding-4B --local-dir ./models/Qwen3-Embedding-4B

python main.py --thinking --rag --limit 300 --generation_seed 42 \
  --embedding_model ./models/Qwen3-Embedding-4B
```
Chroma indexes and history metadata are cached in `.cache/triage_rag/`.

## Evaluate Saved Results

Run the following command in terminal for a basic evluation on a saved triage result file:

```bash
python evaluate.py --result_json triage_results/triage_results_2026-11-07_18-20-04_300_Qwen3-1.7B_thinking.json
```

Metrics cover technical failures/errors, urgency agreement with original priorities, and action agreement with selected reference queues. Action agreement is a partial reference, not overall business correctness. Details in [Technical Document - Metrics and Monitoring](Technical_Document.md#metrics-and-monitoring) Reports are in the folder `evaluation_results`.

### Human Offline Evaluation

Add `--human_review_csv --num_initial_eval 10` to the evaluation command. Modify `--num_initial_eval 10` to control how many tickets you want to evaluate. For the first-time evaluation execution, `triage_human_review_<input suffix>.csv` in the project root is generated and ends with `pending` status. Then you can fill scores for all four `*_eva` columns with integers: **1 unacceptable, 2 mostly unacceptable, 3 partially accepted, 4 mostly accepted, 5 fully accepted**. Rerun the same command and answer `n` to score so that metrics can be automatically calculated and exit/finish the evaluation. Answer `y` to append more tickets if you want. 

Note:Blank or invalid scores can lead to the review pending. The default pass threshold is at least 4 in every dimension.

The included 100-ticket review ([csv_link](triage_human_review_2026-11-07_18-20-04_300_Qwen3-1.7B_thinking.csv)) was scored by an **AI assistant** to replace human effort due to the limit time of this assignment.

## Scenario Tests
Run the command in terminal to start the automatic test
```bash
python -m pytest tests/test_agent_workflow.py -v -s
```
Change `MODEL_PATH`, `THINKING`, and `RAG` in the test file for another configuration. The CSV ([csv_link](tests/scenario_review_20_cases_Qwen3-1.7B_thinking.csv)) in `tests/` supplies empty-input test, five normal, five short, five complex, and five scope-boundary cases. Tests check actions and output consistency. `expected_action` contains test assumptions, not the dataset ground truth labels. The CSV's `actual_*` fields come from saved predictions. Pytest generates fresh outputs. 