import streamlit as st
from rag_engine import setup_rag_pipeline, query_rag
from llm_client import get_llm
llm = get_llm()

# ---- RAG SEARCH TOOL ----
def rag_search_tool(query: str) -> str:
    docs = vectorstore.similarity_search(query, k=4)
    context = "\n\n".join([d.page_content for d in docs])
    return context

# ---- NOTES GENERATION TOOL ----
def generate_notes_tool(topic: str) -> str:
    context = rag_search_tool(topic)
    prompt = f"Make structured study notes on: {topic}\n\nUsing this context:\n{context}"
    return llm.predict(prompt)

# ---- QUIZ GENERATOR TOOL ----
def generate_quiz_tool(topic: str) -> str:
    context = rag_search_tool(topic)
    prompt = f"Generate 5 MCQs with answers based on this context:\n{context}"
    return llm.predict(prompt)

# ---- SUMMARY TOOL ----
def summary_tool(text: str) -> str:
    prompt = f"Summarize this:\n{text}"
    return llm.predict(prompt)

def rag_search_tool(query: str):
    if "qa_chain" not in st.session_state or st.session_state.qa_chain is None:
        return "❌ RAG not initialized. Upload documents first."

    return query_rag(st.session_state.qa_chain, query)

