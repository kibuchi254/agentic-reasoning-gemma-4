from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AgentAction(str, Enum):
    THINK = "think"
    TOOL_CALL = "tool_call"
    RESPOND = "respond"
    ERROR = "error"


class ToolCall(BaseModel):
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class AgentStep(BaseModel):
    iteration: int
    action: AgentAction
    thought: str = ""
    tool_call: ToolCall | None = None
    tool_result: Any | None = None
    observation: str = ""


class AgentResponse(BaseModel):
    id: str
    answer: str
    steps: list[AgentStep] = Field(default_factory=list)
    tool_calls_made: int = 0
    iterations_used: int = 0
    structured_data: dict[str, Any] | None = None


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(system|user|assistant)$")
    content: str


class AgentRequest(BaseModel):
    messages: list[ChatMessage]
    stream: bool = False
    max_tokens: int = Field(2048, ge=1, le=4096)
    temperature: float = Field(0.4, ge=0.0, le=2.0)
    tools: list[str] | None = None
    max_iterations: int = Field(10, ge=1, le=25)
    org_id: str | None = None
    session_id: str | None = None
    response_format: str | None = None


class ChatRequest(BaseModel):
    model: str = ""
    messages: list[dict[str, Any]]
    stream: bool = False
    max_tokens: int = Field(2048, ge=1, le=4096)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    tools: list[Any] | None = None
    tool_choice: Any | None = None

    class Config:
        extra = "allow"


class WorkflowRequest(BaseModel):
    workflow_type: str
    input_data: dict[str, Any]
    org_id: str | None = None
    session_id: str | None = None
