from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from app.agents.engine import AgentEngine
from app.memory.supabase_store import (
    store_agent_trace,
    store_conversation,
    store_workflow_run,
)
from app.models.schemas import AgentRequest, ChatMessage, WorkflowRequest

logger = logging.getLogger("workflows")

WORKFLOW_TEMPLATES: dict[str, dict[str, Any]] = {
    "invoicing": {
        "description": "End-to-end invoicing workflow: create invoice, send notification, track payment",
        "steps": [
            {
                "name": "create_invoice",
                "prompt": "Create an invoice based on the provided data. Use the create_invoice tool with the details from the input.",
            },
            {
                "name": "notify_customer",
                "prompt": "Send a notification to the customer about their new invoice. Include the invoice details in the message.",
            },
        ],
    },
    "customer_onboarding": {
        "description": "New customer onboarding: create CRM record, send welcome, set up billing",
        "steps": [
            {
                "name": "create_crm_record",
                "prompt": "Create a new CRM record for the customer using the update_crm tool.",
            },
            {
                "name": "send_welcome",
                "prompt": "Send a welcome notification to the new customer.",
            },
        ],
    },
    "support_escalation": {
        "description": "Support ticket escalation: analyze issue, search knowledge base, notify team",
        "steps": [
            {
                "name": "search_kb",
                "prompt": "Search the knowledge base for solutions related to the customer's issue.",
            },
            {
                "name": "analyze_and_respond",
                "prompt": "Based on the knowledge base results, provide a solution or recommend escalation. If escalating, notify the support team.",
            },
        ],
    },
    "monthly_report": {
        "description": "Monthly business report: gather metrics, generate report, notify stakeholders",
        "steps": [
            {
                "name": "gather_metrics",
                "prompt": "Query analytics for revenue, customer count, and support ticket metrics for this month.",
            },
            {
                "name": "generate_report",
                "prompt": "Generate a comprehensive monthly report based on the gathered metrics.",
            },
        ],
    },
}


class WorkflowOrchestrator:
    def __init__(self):
        self.engine = AgentEngine()

    async def execute(self, request: WorkflowRequest) -> dict[str, Any]:
        workflow_id = f"wf-{uuid.uuid4().hex[:12]}"
        template = WORKFLOW_TEMPLATES.get(request.workflow_type)

        if not template:
            available = ", ".join(WORKFLOW_TEMPLATES.keys())
            return {
                "workflow_id": workflow_id,
                "status": "error",
                "error": f"Unknown workflow: {request.workflow_type}. Available: {available}",
            }

        await store_workflow_run(
            workflow_id, request.org_id, request.workflow_type, "running"
        )

        results = []
        context = json.dumps(request.input_data)

        for step in template["steps"]:
            agent_req = AgentRequest(
                messages=[
                    ChatMessage(
                        role="system",
                        content=f"You are executing step '{step['name']}' in a '{request.workflow_type}' workflow. Previous results: {json.dumps(results)}",
                    ),
                    ChatMessage(role="user", content=f"{step['prompt']}\n\nInput data: {context}"),
                ],
                max_iterations=5,
                org_id=request.org_id,
                session_id=request.session_id,
            )

            try:
                result = await self.engine.run(agent_req)
                step_result = {
                    "step": step["name"],
                    "answer": result.answer,
                    "tool_calls": result.tool_calls_made,
                    "iterations": result.iterations_used,
                }
                results.append(step_result)
                context = result.answer

                await store_agent_trace(
                    result.id,
                    request.org_id,
                    [s.model_dump() for s in result.steps],
                    result.answer,
                )
            except Exception as exc:
                logger.error("Workflow step %s failed: %s", step["name"], exc)
                results.append({"step": step["name"], "error": str(exc)})

        await store_workflow_run(
            workflow_id,
            request.org_id,
            request.workflow_type,
            "completed",
            {"steps": results},
        )

        return {
            "workflow_id": workflow_id,
            "workflow_type": request.workflow_type,
            "status": "completed",
            "steps": results,
        }

    @staticmethod
    def list_workflows() -> dict[str, Any]:
        return {
            name: {"description": tpl["description"], "steps": [s["name"] for s in tpl["steps"]]}
            for name, tpl in WORKFLOW_TEMPLATES.items()
        }
