import pandas as pd
import pytest
from triage_langgraph import LocalLLM, TicketDataset, TicketHistoryRAG, TicketTriageAgent

###########
scenario_pd = pd.read_csv("./tests/scenario_review_20_cases_Qwen3-1.7B_thinking.csv") #this is AI summarised and expected action labelled for testing purposes. It is used for testing the triage agent's output. So Passed/Failed for single case can not decide the correctness of the triage agent in a insurance background.
MODEL_PATH = "./models/Qwen3-1.7B"
THINKING = False
USE_RAG = False
EMBEDDING_MODEL_PATH = "./models/Qwen3-Embedding-0.6B" #./models/Qwen3-Embedding-4B
##########

normal_cases = scenario_pd[scenario_pd["scenario"] == "normal"][["ticket_id", "expected_action"]].values.tolist()
short_cases = scenario_pd[scenario_pd["scenario"] == "very_short"][["ticket_id", "expected_action"]].values.tolist()
long_complex_cases = scenario_pd[scenario_pd["scenario"] == "long_or_complex"][["ticket_id", "expected_action"]].values.tolist()
oos_cases = scenario_pd[scenario_pd["scenario"] == "support_scope_boundary"][["ticket_id", "expected_action"]].values.tolist()


@pytest.fixture(scope="session")
def agent():
    retriever = None
    if USE_RAG:
        tickets = TicketDataset("./data/aa_dataset-tickets-multi-lang-5-2-50-version.csv").load_ticket(include_history=True)
        excluded_ids = set(scenario_pd['ticket_id'])
        history_rag = TicketHistoryRAG()
        documents = history_rag.prepare_history_documents(tickets, excluded_ids)
        retriever = history_rag.build_retriever(documents, EMBEDDING_MODEL_PATH)
    return TicketTriageAgent(LocalLLM(MODEL_PATH, thinking=THINKING), retriever=retriever)


def triage_ticket(agent, ticket_id, expected_action):
    text = scenario_pd[scenario_pd["ticket_id"] == ticket_id]["text_content"].iloc[0] #get the text subject+body
    result = agent.triage({"ticket_id": ticket_id, "text": text, "language": ""})
    print(result)

    if USE_RAG:
        if text.strip():
            assert result['retrieved_ticket_ids'], 'RAG returned no historical tickets'
            assert set(result['retrieved_ticket_ids']).isdisjoint(scenario_pd['ticket_id'])

    #check basic status and output fields
    assert result["ticket_id"] == ticket_id
    assert result["status"] != "error", result["notes"]
    assert result["notes"].strip()

    if result["next_action"] is None:
        assert result["status"] == "needs_more_info"
        assert result.get("clarification_question", "").strip()
    elif result["next_action"] == "Escalate to human supervisor":
        assert result["status"] == "needs_review"
    else:
        assert result["status"] == "ok"

    action = "clarify" if result["next_action"] is None else result["next_action"]
    expected_action_processed = expected_action.split("|") #optional, as the generated scenario_review_20_cases_Qwen3-1.7B_thinking.csv has some multiple expected actions.
    allowed_actions = []
    for eap in expected_action_processed:
        allowed_actions.append(eap.strip())
    assert action in allowed_actions, result #keep result foor debugging if the test fails

    return result


def test_empty_ticket():
    empty_agent = TicketTriageAgent(llm=None)
    result = empty_agent.triage({"ticket_id": -1, "text": "", "language": ""})

    assert result["ticket_id"] == -1
    assert result["status"] == "needs_more_info"
    assert result["next_action"] is None
    assert result["clarification_question"].strip()


@pytest.mark.parametrize("ticket_id, expected_action", normal_cases)
def test_normal_ticket(agent, ticket_id, expected_action):
    triage_ticket(agent, ticket_id, expected_action)


@pytest.mark.parametrize("ticket_id, expected_action", short_cases)
def test_short_ticket(agent, ticket_id, expected_action):
    triage_ticket(agent, ticket_id, expected_action)


@pytest.mark.parametrize("ticket_id, expected_action", long_complex_cases)
def test_long_complex_ticket(agent, ticket_id, expected_action):
    triage_ticket(agent, ticket_id, expected_action)


@pytest.mark.parametrize("ticket_id, expected_action", oos_cases)
def test_support_scope_boundary(agent, ticket_id, expected_action):
    triage_ticket(agent, ticket_id, expected_action)

#running example: python -m pytest tests/test_agent_workflow.py -v -s

"""
eg: logic difference
E       AssertionError: {'ticket_id': 23712, 'text_content': 'Subject: Support for Digital Strategies to Enhance Brand Growth
E         Body: Reach out to us to explore digital strategies that can help substantiate brand growth. We can provide detailed insights into methods for increasing brand awareness and customer engagement. We are also ready to share information on social mediamarketing, content creation, and campaign management. Additionally, we can offer a deeper understanding of performance metrics that reflect market changes, including case studies and demonstrating the effectiveness of various strategies.', 'next_action': 'Send standard FAQ or self-service link', 'status': 'ok', ...}
E       assert 'Send standard FAQ or self-service link' in ['clarify', 'Escalate to human supervisor']

tests/test_agent_workflow.py:59: AssertionError
"""