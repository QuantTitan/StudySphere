import streamlit as st
from rag_engine import query_rag

def rag_search_tool(query: str) -> str:
    """Simple helper for non-agent code paths to query session RAG."""
    if "qa_chain" not in st.session_state or st.session_state.qa_chain is None:
        return "❌ RAG not initialized. Upload documents first."
    return query_rag(st.session_state.qa_chain, query)

def generate_notes_tool(topic: str) -> str:
    context = rag_search_tool(topic)
    return f"Notes generation requires an LLM. Use the agent (created after entering API key) to produce notes.\n\nContext:\n{context}"

def generate_quiz_tool(topic: str) -> str:
    context = rag_search_tool(topic)
    return f"Quiz generation requires an LLM. Use the agent (created after entering API key).\n\nContext:\n{context}"

def summary_tool(text: str) -> str:
    return f"Summary requires an LLM. Use the agent (created after entering API key).\n\nText:\n{text}"
