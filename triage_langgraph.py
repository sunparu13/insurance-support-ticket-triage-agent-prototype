from langgraph.graph import StateGraph, START, END
from typing import Annotated, Literal, TypedDict
import os
import json
import torch
import csv
import hashlib
import random

from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoModelForCausalLM, AutoTokenizer
from prompt import CLASSIFICATION_PROMPT, HISTORY_CONTEXT_PROMPT, MISSING_INFO_PROMPT, NEXT_ACTION_PROMPT

MAX_INPUT_TOKENS = 8192  #40960 for qwen3 1.7b, for us, extreme case can be around 2500
MAX_OUTPUT_TOKENS = 2048
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "Qwen3-1.7B")

TOPICS = [
    "Policy / Contract",
    "Claims / Damage",
    "Billing / Payment",
    "Technical / Online Access",
    "Account / Customer Details",
    "Product / Service Feedback",
    "Other",
]

URGENCIES = ["Low", "Medium", "High"]

ACTIONS = [
    "Send standard FAQ or self-service link",
    "Create or update a claim",
    "Forward to billing team",
    "Forward to technical support",
    "Escalate to human supervisor"
]

class TicketDataset:
    """
    data loader and normalizer
    """

    def __init__(self, csv_path: str):
        self.csv_path=csv_path

    def normalize_text(self, value: str | None) -> str:
        """
        normalize line endings and trim edges without changing language/content
        """
        return (value or "").replace("\r\n", "\n").replace("\r", "\n").strip()

    def load_ticket(self, include_history=False) -> list[dict]:
        with open(self.csv_path, 'r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            tickets = []
            for record_id, row in enumerate(reader):
                subject = self.normalize_text(row.get("subject"))
                body = self.normalize_text(row.get("body"))

                if subject or body:
                    text = f"Subject: {subject}\nBody: {body}" 
                else:
                    text = ""

                ticket = {
                    'ticket_id': record_id,
                    'text': text,
                    'language': row.get("language") or "",
                }
                if include_history:
                    ticket['priority'] = row.get('priority') or ''
                    ticket['queue'] = row.get('queue') or ''
                    ticket['type'] = row.get('type') or ''
                    tags = []
                    for column in row:
                        if column.startswith('tag_') and row[column]:
                            tags.append(row[column])
                    ticket['tags'] = tags
                tickets.append(ticket)
        return tickets


class TicketHistoryRAG:
    """
    the rag part,including:
    1. prepare_history_documents: exclude evaluation IDs and randomly sample historical tickets.
    2. build_retriever: build or reuse a Chroma index for the supplied historical documents.
    """
    def prepare_history_documents(self, tickets, excluded_ids, count=2000, seed=42):
        history = []
        for ticket in tickets:
            if ticket['ticket_id'] not in excluded_ids and ticket['text'].strip(): #do not include blank and triage tickets
                history.append(ticket)

        sampled = random.Random(seed).sample(history, count)

        documents = []
        for ticket in sampled:
            metadata = ticket.copy()
            text = metadata.pop('text')
            metadata['tags'] = json.dumps(metadata['tags'])
            documents.append(Document(page_content=text, metadata=metadata))
        return documents

    def build_retriever(self, documents, embedding_model, top_k=3, cache_dir='.cache/triage_rag'):
        if not documents:
            raise ValueError('Historical documents must not be empty')
        dtype = torch.float16 if torch.cuda.is_available() or torch.backends.mps.is_available() else torch.float32 #same as LocalLLM

        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'local_files_only': True, 'model_kwargs': {'dtype': dtype}},
            encode_kwargs={'batch_size': 8, 'prompt': '', 'normalize_embeddings': True},
            query_encode_kwargs={'prompt_name': 'query', 'normalize_embeddings': True},
        )

        chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(documents)
        signature = json.dumps({'model': os.path.realpath(embedding_model),
                                'encode': embeddings.encode_kwargs, 'query_encode': embeddings.query_encode_kwargs,
                                'chunks': [doc.model_dump() for doc in chunks]}, sort_keys=True)
        directory = os.path.join(cache_dir, hashlib.sha256(signature.encode()).hexdigest()[:16])
        settings = Settings(anonymized_telemetry=False)
        store = Chroma(collection_name='ticket_history', embedding_function=embeddings,
                       persist_directory=directory, client_settings=settings)
        if store._collection.count() != len(chunks):
            store.delete_collection()
            store = Chroma.from_documents(chunks, embeddings, collection_name='ticket_history',
                                          persist_directory=directory, client_settings=settings,
                                          ids=[str(i) for i in range(len(chunks))]) #vectorize creation
        print(f'History: {len(documents)} tickets; index: {directory}')
        return store.as_retriever(search_kwargs={'k': min(len(chunks), top_k * 8)})


class TriageState(TypedDict):
    ticket: dict
    initial_assessment: dict
    result: dict

class InferenceLimitError(ValueError):
    """Raised when input or output exceeds the token budget"""

class ModelOutputError(ValueError):
    """Raised when the model's output is invalid"""

class LocalLLM:
    """
    loads a local LLM model, config model params, and generate JSON output from a prompt
    The input and output token limit are checked.
    """

    def __init__(self, model_path=None, thinking=False):
        self.thinking = thinking
        model_path = os.path.expanduser(model_path or DEFAULT_MODEL_PATH)
        if not os.path.isdir(model_path):
            raise FileNotFoundError("Cannot find model folder: " + model_path)
        with open(os.path.join(model_path, "config.json"), encoding="utf-8") as f:
            model_type = json.load(f)["model_type"]
        if model_type not in {"qwen3", "qwen3_5", "qwen3_5_text"}:
            raise ValueError(f"Unsupported model type: {model_type}")
        if not thinking:
            temperature, top_p = 0.7, 0.8
        elif model_type == "qwen3":
            temperature, top_p = 0.6, 0.95
        else:
            temperature, top_p = 1.0, 0.95
        self.generation_params = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": 20,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
        }
        if torch.backends.mps.is_available():
            device = "mps"
            dtype = torch.float16
        elif torch.cuda.is_available():
            device = "cuda"
            dtype = torch.float16
        else:
            device = "cpu"
            dtype = torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True, dtype=dtype,
        )
        self.model.to(device)
        self.model.eval()

    def generate_json(self, messages: list[dict]) -> dict:
        """Generate a response, remove thinking tokens, and parse a JSON object."""
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=self.thinking,
            return_dict=True,
            return_tensors="pt",
        )
        input_length = inputs["input_ids"].shape[1]
        if input_length > MAX_INPUT_TOKENS:
            raise InferenceLimitError("Input exceeds the token budget")

        output_budget = 8192 if self.thinking else MAX_OUTPUT_TOKENS
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs.to(self.model.device),
                max_new_tokens=output_budget,
                do_sample=True,
                **self.generation_params,
            )

        answer_tokens = outputs[0, input_length:].tolist()
        output_length = len(answer_tokens)
        if self.thinking:
            end_thinking_id = self.tokenizer.convert_tokens_to_ids("</think>")
            if end_thinking_id not in answer_tokens:
                if output_length >= output_budget:
                    raise InferenceLimitError("Thinking did not finish within the output budget")
                raise ModelOutputError("Thinking response is missing </think>")
            answer_tokens = answer_tokens[answer_tokens.index(end_thinking_id) + 1:]

        answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
        try:
            result = json.loads(answer)
        except json.JSONDecodeError as error:
            if output_length >= output_budget:
                raise InferenceLimitError("Final JSON did not finish within the output budget") from error
            raise ModelOutputError("Final answer must be valid JSON") from error

        if not isinstance(result, dict):
            raise ModelOutputError("Model response must be a JSON object")
        return result

class TicketTriageAgent:
    """
    core class. It finishes all triage steps and is orchastrated by langgraph state graph.
    """
    def __init__(self, llm, retriever=None, rag_top_k=3):
        self.llm = llm
        self.retriever = retriever
        self.rag_top_k = rag_top_k
        self.graph= self.build_graph()

    def retrieve_history(self, text):
        """return top-k history ticket with one truck per ticket_id"""
        retrieved_chunks = self.retriever.invoke(text)
        history = []
        selected_ticket_ids = []

        for chunk in retrieved_chunks:
            ticket_id = chunk.metadata['ticket_id']
            if ticket_id in selected_ticket_ids:
                continue #avoid duplicate tickets

            history_ticket = chunk.metadata.copy()
            history_ticket['tags'] = json.loads(history_ticket['tags'])
            history_ticket['text'] = chunk.page_content

            history.append(history_ticket)
            selected_ticket_ids.append(ticket_id)
            if len(history) == self.rag_top_k:
                break
        return history

    def classify_and_prioritize(self, text: str, history=None) -> dict:
        """
        classify the topic, judge urgency and identify the next workflow branch
        text: the ticket text to be classified and prioritized
    returns:
        history: a list of historical tickets retrieved for context
        result: a dictionary containing the classification result
        with keys 'topic', 'urgency', 'decision', and 'reason'
        """
        system = CLASSIFICATION_PROMPT
        content = text
        if history:
            system += HISTORY_CONTEXT_PROMPT
            #content = json.dumps({'ticket_text': text, 'retrieved_history': history}, ensure_ascii=False)
            content = json.dumps({'retrieved_history': history,'ticket_text': text,}, ensure_ascii=False)
        result = self.llm.generate_json([
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]) 
        return self.validate_classification(result) #validate by raise error

    def routing_condition(self, initial_assessment: dict)-> str:
        """
        based on the classification result, judge the
        so called "intermediate" routing condition
        """
        return initial_assessment['decision']


    def check_missing_info(self, text: str, missing_info: str) -> str:
        """
        generate clarification questions based on the missing information.
        """
        ticket_context={
            "ticket_text": text,
            "missing_info": missing_info
        }
        result = self.llm.generate_json([
            {"role": "system", "content": MISSING_INFO_PROMPT},
            {"role": "user", "content": json.dumps(ticket_context, ensure_ascii=False)},
        ])
        if not isinstance(result, dict) or "question" not in result:
            raise ModelOutputError("clarification question result must be a JSON object")
        question = result.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ModelOutputError("Missing or empty clarification question")
        return question.strip()



    def recommend_next_action(self, text: str, topic: str, urgency: str, classification_reason: str) -> dict:
        """
        recommend the next action based on the ticket text, topic, urgency and classification reason.
        """
        if topic not in TOPICS or urgency not in URGENCIES:
            raise ModelOutputError("valid topic and urgency are required for next action recommendation")
        ticket_context = {'ticket_text': text, 
                          'topic': topic, 
                          'urgency': urgency,
                          'classification_reason': classification_reason,
                         }
        result = self.llm.generate_json([
            {"role": "system", "content": NEXT_ACTION_PROMPT},
            {"role": "user", "content": json.dumps(ticket_context, ensure_ascii=False)},
        ])
        
        if not isinstance(result, dict):
            raise ModelOutputError("recommendation must be a JSON object")
        action = result.get("next_action")
        if not isinstance(action, str) or action not in ACTIONS:
            raise ModelOutputError("next_action must be one in allowed_actions")
        notes = result.get("notes")
        if not isinstance(notes, str) or not notes.strip():
            raise ModelOutputError("notes missing or empty")
        return {"next_action": action, "notes": notes.strip()}

    def triage(self, ticket):
        text = ticket['text']
        if not isinstance(text, str):
            raise TypeError("ticket['text'] must be a string")
        initial_state = {
            "ticket": ticket,
            "initial_assessment": {},
            "result": {
                "ticket_id": ticket["ticket_id"],
                "text_content": text,
                "next_action": None,
                "status": None,
                "notes": "",
            },
        }
        final_state = self.graph.invoke(initial_state) #executes the state graph
        assessment = final_state['initial_assessment']
        result = final_state['result']
        result['topic'] = assessment.get('topic')
        result['urgency'] = assessment.get('urgency')
        result['decision'] = assessment.get('decision')
        result['classification_reason'] = assessment.get('reason')
        return result

    #########graph adaptation#############

    def classification_node(self, state: TriageState):
        text = state['ticket']['text']
        result = state['result'].copy()
        if not text.strip():
            #blank ticket text, no LLM inference, just to end and gives fixed clarification question
            result.update(
                next_action=None,
                status="needs_more_info",
                notes="No issue was provided. What problem do you need help with?",
                clarification_question="What problem do you need help with?",
            )
            return {'result': result}

        history = []
        if self.retriever:
            history = self.retrieve_history(text)
            result['retrieved_ticket_ids'] = []
            for ticket in history:
                result['retrieved_ticket_ids'].append(ticket['ticket_id'])

        initial_assessment = self.classify_and_prioritize(text, history)
        return {'initial_assessment': initial_assessment, 'result': result}

    def choose_next(self, state: TriageState) -> str:
            """
            2.based on the classification result, judge the
            so called "intermediate" routing condition
            """
            if not state['initial_assessment']:
                return 'end'
            return self.routing_condition(state['initial_assessment'])

    def missing_info_node(self, state: TriageState):
        try:
            question = self.check_missing_info(
                state["ticket"]["text"], state["initial_assessment"]["reason"],
            )
            result = state['result'].copy()
            result.update(next_action=None, 
                          status='needs_more_info', 
                          clarification_question=question,
                          notes=question)
            return {'result': result}
        except (ValueError, RuntimeError) as error:
            return self.error_update(state, 'missing_info_check', error)

    def human_review_node(self, state: TriageState):
        initial_assessment = state['initial_assessment']
        result = state['result'].copy()
        result['next_action'] = 'Escalate to human supervisor'
        result['status'] = 'needs_review'    #status can be 'ok', 'needs_review', 'needs_more_info', or 'error'
        result['notes'] = initial_assessment['reason']
        return {'result': result}

    
    def next_action_node(self, state: TriageState):
        try:
            initial_assessment = state['initial_assessment']
            recommendation = self.recommend_next_action(
                state["ticket"]["text"], 
                initial_assessment["topic"], 
                initial_assessment["urgency"],
                initial_assessment["reason"]
            )
            result = state['result'].copy()
            result.update(recommendation)
            if result['next_action'] == 'Escalate to human supervisor':
                result['status'] = 'needs_review'
            else:
                result['status'] = 'ok'
            return {'result': result}
        except (ValueError, RuntimeError) as error:
            return self.error_update(state, 'recommend_next_action', error)
        
    
    def validate_classification(self, result):
        if not isinstance(result, dict) or not all(key in result for key in ('topic', 'urgency', 'decision', 'reason')):
            raise ModelOutputError("Classification result must be a JSON object with keys: topic, urgency, decision, reason")
        result = result.copy()
        if isinstance(result["decision"], str):
            result["decision"] = result["decision"].strip().lower() #get Review output in experiemnt, please notice lower
        if result['topic'] is not None and result['topic'] not in TOPICS:
            raise ModelOutputError(f"Invalid topic: {result['topic']}")
        if result['urgency'] is not None and result['urgency'] not in URGENCIES:
            raise ModelOutputError(f"Invalid urgency: {result['urgency']}")
        if result['decision'] not in ['recommend', 'clarify', 'review']:
            raise ModelOutputError(f"Invalid decision: {result['decision']}")
        reason = result['reason'] #show why recommend/clarify/review are judged.
        if not isinstance(reason, str) or not reason.strip():
            raise ModelOutputError('Classification reason must be exist')
        if result['decision'] == 'recommend': #make sure recommend route should have topic and urgency info
            if result['topic'] is None or result['urgency'] is None:
                raise ModelOutputError('Cannot proceed without topic and urgency')
        return {
            
            'topic': result['topic'],
            'urgency': result['urgency'],
            'decision': result['decision'],
            'reason': reason.strip()
        }
        

    def error_update(self, state: TriageState, node_name: str, error: Exception):
        """
        error handling when runtime error occurs, inference limit error occurs, or model output is invalid.
        args:
            state: the current state of the triage process
            node_name: the name of the node where the error occurred, e.g., 'classify_and_prioritize', 'missing_info_check', or 'recommend_next_action'
            error: the exception that was raised
        """
        result = state['result'].copy()
        result['next_action'] = 'Escalate to human supervisor' #or just write None
        result['status'] = 'error'
        result['error_stage'] = node_name
        if isinstance(error, InferenceLimitError):
            result['error_type'] = 'token_budget'
        elif isinstance(error, RuntimeError): #model runtime error
            result['error_type'] = 'inference_runtime_error'
        else:
            result['error_type'] = 'invalid_output'
        result['notes'] = f"Error in {node_name}: {str(error)}"
        return {'result': result}

    def build_graph(self):
            """
            builds the state graph for the ticket triage process, defining nodes and edges 
            for classification, human review, missing information check, and next action recommendation.
            """
            graph= StateGraph(TriageState)
            graph.add_node('classify_and_prioritize', self.classification_node)
            graph.add_node('human_review', self.human_review_node)
            graph.add_node('missing_info_check', self.missing_info_node)
            graph.add_node('recommend_next_action', self.next_action_node)
    
            graph.add_edge(START, 'classify_and_prioritize')
            graph.add_conditional_edges("classify_and_prioritize", self.choose_next, {
                "end": END,
                "review": "human_review",
                "clarify": "missing_info_check",
                "recommend": "recommend_next_action",
            })
            graph.add_edge('human_review', END)
            graph.add_edge('missing_info_check', END)
            graph.add_edge('recommend_next_action', END)
            return graph.compile()









