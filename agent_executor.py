from langchain.agents import initialize_agent, AgentType, Tool
from llm_client import get_llm
import streamlit as st
from rag_engine import query_rag

def create_agent(api_key: str):
    """Create and return a LangChain agent bound to a Gemini LLM using api_key."""
    llm = get_llm(api_key)

    # tool implementations close over llm and use session_state.qa_chain
    def rag_search_tool_func(query: str) -> str:
        if "qa_chain" not in st.session_state or st.session_state.qa_chain is None:
            return "❌ RAG not initialized. Upload documents first."
        return query_rag(st.session_state.qa_chain, query)

    def generate_notes_tool_func(topic: str) -> str:
        context = rag_search_tool_func(topic)
        prompt = f"Make structured study notes on: {topic}\n\nUsing this context:\n{context}"
        # llm.predict may vary by LLM wrapper; using predict for LangChain model wrapper
        return llm.predict(prompt)

    def generate_quiz_tool_func(topic: str) -> str:
        context = rag_search_tool_func(topic)
        prompt = f"Generate 5 MCQs with answers based on this context:\n{context}"
        return llm.predict(prompt)

    def summary_tool_func(text: str) -> str:
        prompt = f"Summarize this:\n{text}"
        return llm.predict(prompt)

    tools = [
        Tool(name="RAG_Search", func=rag_search_tool_func,
             description="Retrieve content from uploaded PDFs."),
        Tool(name="Generate_Notes", func=generate_notes_tool_func,
             description="Generate structured notes for a topic."),
        Tool(name="Generate_Quiz", func=generate_quiz_tool_func,
             description="Generate 5 MCQs with answers for a topic."),
        Tool(name="Summarize", func=summary_tool_func,
             description="Summarize provided text.")
    ]

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    return agent