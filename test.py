import json
import time

import requests

BASE = "http://localhost:8000"


def test_health():
    r = requests.get(f"{BASE}/health")
    print("=== Health ===")
    print(json.dumps(r.json(), indent=2))
    assert r.status_code == 200


def test_simple_chat():
    r = requests.post(f"{BASE}/v1/chat/completions", json={
        "model": "gemma4:latest",
        "messages": [
            {"role": "system", "content": "You are a business AI assistant."},
            {"role": "user", "content": "What are the key metrics for SaaS businesses?"},
        ],
    })
    print("\n=== Simple Chat ===")
    data = r.json()
    print(data["choices"][0]["message"]["content"][:300])
    print(f"Tokens: {data['usage']}")


def test_agent_run():
    r = requests.post(f"{BASE}/v1/agent/run", json={
        "messages": [
            {"role": "user", "content": "Create an invoice for customer CUST-001 with 2 items: Web Development ($5000) and SEO Setup ($2000). Due in 30 days."},
        ],
        "max_iterations": 5,
        "session_id": "test-session-001",
    })
    print("\n=== Agent Run ===")
    data = r.json()
    print(f"Agent ID: {data['id']}")
    print(f"Answer: {data['answer'][:300]}")
    print(f"Tool calls: {data['tool_calls_made']}")
    print(f"Iterations: {data['iterations_used']}")
    print(f"Steps: {len(data['steps'])}")
    for step in data["steps"]:
        print(f"  Step {step['iteration']}: {step['action']} — {step.get('thought', '')[:100]}")


def test_agent_crm():
    r = requests.post(f"{BASE}/v1/agent/run", json={
        "messages": [
            {"role": "user", "content": "Search for customer John Doe in the CRM and show their history."},
        ],
        "max_iterations": 5,
    })
    print("\n=== Agent CRM ===")
    data = r.json()
    print(f"Answer: {data['answer'][:300]}")
    print(f"Tool calls: {data['tool_calls_made']}")


def test_workflow():
    r = requests.post(f"{BASE}/v1/workflow/list")
    print("\n=== Available Workflows ===")
    print(json.dumps(r.json(), indent=2))

    r = requests.post(f"{BASE}/v1/workflow/run", json={
        "workflow_type": "invoicing",
        "input_data": {
            "customer_id": "CUST-042",
            "items": [
                {"description": "Consulting Hours", "quantity": 10, "unit_price": 150},
                {"description": "Cloud Hosting Setup", "quantity": 1, "unit_price": 500},
            ],
            "due_date": "2026-06-15",
        },
    })
    print("\n=== Workflow Run ===")
    data = r.json()
    print(f"Workflow ID: {data['workflow_id']}")
    print(f"Status: {data['status']}")
    for step in data.get("steps", []):
        print(f"  Step: {step['step']} — {step.get('answer', step.get('error', ''))[:200]}")


def test_tools():
    r = requests.get(f"{BASE}/v1/tools")
    print("\n=== Tools ===")
    data = r.json()
    print(f"Registered: {data['count']} tools")
    for t in data["tools"]:
        print(f"  - {t['name']}: {t['description']}")


if __name__ == "__main__":
    print("Testing Gemma Agentic Business AI...\n")
    test_health()
    test_tools()
    test_simple_chat()
    test_agent_run()
    test_agent_crm()
    test_workflow()
    print("\nAll tests complete.")
