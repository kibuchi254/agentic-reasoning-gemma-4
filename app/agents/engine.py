from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any

from openai import AsyncOpenAI

from app.config import (
    AGENT_MAX_ITERATIONS,
    AGENT_TEMPERATURE,
    MAX_NEW_TOKENS,
    MAX_PROMPT_TOKENS,
    MODEL,
    OLLAMA_API_KEY,
    OLLAMA_BASE_URL,
)
from app.models.schemas import (
    AgentAction,
    AgentRequest,
    AgentResponse,
    AgentStep,
    ToolCall,
)
from app.tools.registry import ToolRegistry

logger = logging.getLogger("agent_engine")

REACT_SYSTEM_PROMPT = """You are a highly capable business AI assistant with agentic reasoning abilities.

You solve tasks by thinking step-by-step and using tools when needed. Follow this exact format:

THOUGHT: <your reasoning about what to do next>
ACTION: <tool_name> | <json arguments>  (OR "respond" if you have the final answer)
OBSERVATION: <result from tool, filled automatically>

When you have enough information to give a final answer, use:
THOUGHT: <final reasoning>
ACTION: respond
FINAL_ANSWER: <your complete answer to the user>

Rules:
- Always think before acting
- Use tools when you need external data or actions
- You may use multiple tool calls in sequence
- Be precise with JSON arguments
- If a tool fails, try an alternative approach
- When responding, be thorough and actionable

Available tools:
{tools_description}"""


def _parse_action(text: str) -> tuple[AgentAction, str, ToolCall | None]:
    action_match = re.search(r"ACTION:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if not action_match:
        return AgentAction.THINK, text, None

    action_str = action_match.group(1).strip()

    if action_str.lower() == "respond":
        final_match = re.search(r"FINAL_ANSWER:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
        answer = final_match.group(1).strip() if final_match else ""
        return AgentAction.RESPOND, answer, None

    parts = action_str.split("|", 1)
    tool_name = parts[0].strip()

    args = {}
    if len(parts) > 1:
        try:
            args = json.loads(parts[1].strip())
        except json.JSONDecodeError:
            args = {"raw_input": parts[1].strip()}

    thought_match = re.search(r"THOUGHT:\s*(.+?)(?:\nACTION:)", text, re.IGNORECASE | re.DOTALL)
    thought = thought_match.group(1).strip() if thought_match else ""

    return AgentAction.TOOL_CALL, thought, ToolCall(tool_name=tool_name, arguments=args)


class AgentEngine:
    def __init__(self):
        self.client = AsyncOpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key=OLLAMA_API_KEY,
            timeout=120.0,
            max_retries=2,
        )
        self.registry = ToolRegistry()

    async def run(self, request: AgentRequest) -> AgentResponse:
        request_id = f"agent-{uuid.uuid4().hex[:12]}"
        steps: list[AgentStep] = []
        tool_calls_made = 0

        tools_desc = self.registry.describe(request.tools)
        system_prompt = REACT_SYSTEM_PROMPT.format(tools_description=tools_desc)

        messages = [{"role": "system", "content": system_prompt}]
        for m in request.messages:
            messages.append({"role": m.role, "content": m.content})

        max_iter = request.max_iterations or AGENT_MAX_ITERATIONS
        temperature = request.temperature or AGENT_TEMPERATURE
        max_tokens = request.max_tokens or MAX_NEW_TOKENS

        for i in range(max_iter):
            try:
                resp = await self.client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except Exception as exc:
                logger.error("[%s] LLM call failed at iter %d: %s", request_id, i, exc)
                steps.append(AgentStep(
                    iteration=i,
                    action=AgentAction.ERROR,
                    thought=str(exc),
                ))
                break

            content = resp.choices[0].message.content or ""
            action, thought, tool_call = _parse_action(content)

            if action == AgentAction.RESPOND:
                steps.append(AgentStep(
                    iteration=i,
                    action=AgentAction.RESPOND,
                    thought=thought,
                    observation=thought,
                ))
                return AgentResponse(
                    id=request_id,
                    answer=thought,
                    steps=steps,
                    tool_calls_made=tool_calls_made,
                    iterations_used=i + 1,
                )

            if action == AgentAction.TOOL_CALL and tool_call:
                tool_result = await self._execute_tool(tool_call)
                tool_calls_made += 1

                steps.append(AgentStep(
                    iteration=i,
                    action=AgentAction.TOOL_CALL,
                    thought=thought,
                    tool_call=tool_call,
                    tool_result=tool_result,
                    observation=str(tool_result),
                ))

                messages.append({"role": "assistant", "content": content})
                obs_text = json.dumps(tool_result, default=str) if isinstance(tool_result, (dict, list)) else str(tool_result)
                messages.append({
                    "role": "user",
                    "content": f"OBSERVATION: {obs_text}\n\nContinue with your reasoning.",
                })
                continue

            steps.append(AgentStep(
                iteration=i,
                action=AgentAction.THINK,
                thought=content,
            ))
            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": "Please continue reasoning. Use ACTION: respond when you have the final answer.",
            })

        try:
            resp = await self.client.chat.completions.create(
                model=MODEL,
                messages=messages + [{"role": "user", "content": "Provide your final answer now based on all reasoning above."}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            final_answer = resp.choices[0].message.content or ""
        except Exception:
            final_answer = "I was unable to complete the full reasoning chain. Please try again."

        return AgentResponse(
            id=request_id,
            answer=final_answer,
            steps=steps,
            tool_calls_made=tool_calls_made,
            iterations_used=max_iter,
        )

    async def _execute_tool(self, tool_call: ToolCall) -> Any:
        tool = self.registry.get(tool_call.tool_name)
        if not tool:
            return {"error": f"Unknown tool: {tool_call.tool_name}"}
        try:
            return await tool.execute(tool_call.arguments)
        except Exception as exc:
            logger.error("Tool %s failed: %s", tool_call.tool_name, exc)
            return {"error": str(exc)}
