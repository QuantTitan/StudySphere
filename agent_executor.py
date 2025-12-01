from langchain.agents import initialize_agent, AgentType, Tool
from llm_client import get_llm
import streamlit as st
from rag_engine import query_rag
import time
from observability import tracker

# add imports for existing tool functions
from agent_tools import (
    rag_search_tool,
    generate_notes_tool,
    generate_quiz_tool,
    summary_tool
)

# Role-Enriched System Prompt 
SYSTEM_PROMPT = """You are StudySphere AI-Tutor, an expert educational assistant specialized in providing accurate, contextually-rich educational guidance.

**Your Core Responsibilities:**
1. Answer questions based on (but not limited to) uploaded course materials
2. Cite specific sources (file names and page numbers) for all claims
3. Provide structured, clear explanations with examples when relevant
4. Identify knowledge gaps and suggest further reading from materials
5. Adapt complexity level to match student needs

**Tools Available:**
- RAG_Search: Retrieve content from course PDFs
- Generate_Notes: Create structured study notes
- Generate_Quiz: Generate practice questions with answers
- Summarize: Condense complex topics

**Response Guidelines:**
- Include relevant examples from course materials 
- Flag when information is NOT covered in materials (but do not limit responses to only those)
- Encourage critical thinking with follow-up questions
- Make sure to highlight cited sources clearly
- Include equations or formulas when applicable

You are committed to fostering deep understanding and academic integrity."""

def _make_agent(llm, tools, system_message=None):
    """Helper to create an agent from an llm and list of tools."""
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        agent_kwargs={"system_message": system_message or SYSTEM_PROMPT}
    )
    return agent

def create_rag_agent(api_key: str):
    """Agent focused on retrieval-only (returns excerpts and sources)."""
    llm = get_llm(api_key)
    tools = [
        Tool(name="RAG_Search", func=rag_search_tool,
             description="Retrieve content from uploaded PDFs with citations.")
    ]
    return _make_agent(llm, tools, system_message="RAG Retrieval Agent — return exact excerpts and source citations.")

def create_tutor_agent(api_key: str):
    """Tutor agent for Q&A using RAG and summary tool."""
    llm = get_llm(api_key)
    tools = [
        Tool(name="RAG_Search", func=rag_search_tool,
             description="Retrieve content from uploaded PDFs with citations."),
        Tool(name="Summarize", func=summary_tool,
             description="Summarize or simplify retrieved content.")
    ]
    return _make_agent(llm, tools)

def create_notes_agent(api_key: str):
    """Notes generator agent — builds structured study notes from retrieved context."""
    llm = get_llm(api_key)
    tools = [
        Tool(name="Generate_Notes", func=generate_notes_tool,
             description="Create structured notes from course materials."),
        Tool(name="RAG_Search", func=rag_search_tool,
             description="Optional retrieval helper for notes generation.")
    ]
    return _make_agent(llm, tools)

def create_summarizer_agent(api_key: str):
    """Standalone summarizer agent."""
    llm = get_llm(api_key)
    tools = [
        Tool(name="Summarize", func=summary_tool,
             description="Condense complex topics into concise summaries.")
    ]
    return _make_agent(llm, tools)

def wrapped_agent_run(agent, prompt, agent_name):
    """Run agent and track latency"""
    start_time = time.time()
    response = agent.run(prompt)
    latency = time.time() - start_time
    tracker.log_prompt_latency(agent_name, latency)
    tracker.log_query(prompt, agent_name, response, latency)
    return response