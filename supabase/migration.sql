-- Gemma Agentic Business AI - Supabase Tables
-- Run this in your Supabase SQL Editor

-- Conversations storage
CREATE TABLE IF NOT EXISTS ai_conversations (
    id BIGSERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    org_id TEXT,
    role TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant')),
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_conversations_session ON ai_conversations(session_id);
CREATE INDEX idx_conversations_org ON ai_conversations(org_id);

-- Agent reasoning traces
CREATE TABLE IF NOT EXISTS ai_agent_traces (
    id BIGSERIAL PRIMARY KEY,
    agent_id TEXT NOT NULL,
    org_id TEXT,
    steps JSONB DEFAULT '[]',
    final_answer TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_agent_traces_agent ON ai_agent_traces(agent_id);
CREATE INDEX idx_agent_traces_org ON ai_agent_traces(org_id);

-- Workflow execution history
CREATE TABLE IF NOT EXISTS ai_workflow_runs (
    id BIGSERIAL PRIMARY KEY,
    workflow_id TEXT NOT NULL UNIQUE,
    org_id TEXT,
    workflow_type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    result JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_workflow_runs_org ON ai_workflow_runs(org_id);
CREATE INDEX idx_workflow_runs_type ON ai_workflow_runs(workflow_type);

-- Row Level Security (enable for multi-tenant)
ALTER TABLE ai_conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE ai_agent_traces ENABLE ROW LEVEL SECURITY;
ALTER TABLE ai_workflow_runs ENABLE ROW LEVEL SECURITY;

-- Example RLS policy (adjust based on your auth setup)
-- CREATE POLICY "Users can see own org data" ON ai_conversations
--     FOR ALL USING (org_id = auth.jwt() ->> 'org_id');
