from langchain.agents import initialize_agent, AgentType, Tool
from langchain.agents.agent import AgentExecutor
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from llm_client import get_llm
import streamlit as st
from rag_engine import query_rag

# Role-Enriched System Prompt for stronger agent control
SYSTEM_PROMPT = """You are StudySphere AI-Tutor, an expert educational assistant specialized in providing accurate, contextually-rich educational guidance.

**Your Core Responsibilities:**
1. Answer questions exclusively based on uploaded course materials
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
- Always prioritize accuracy over completeness
- Use simple language unless technical terms are necessary
- Include relevant examples from course materials
- Flag when information is NOT covered in materials
- Encourage critical thinking with follow-up questions

You are committed to fostering deep understanding and academic integrity."""

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
        prompt = f"Based on the course materials, create structured study notes on: {topic}\n\nContext:\n{context}\n\nProvide clear headings, bullet points, and key takeaways."
        return llm.predict(prompt)

    def generate_quiz_tool_func(topic: str) -> str:
        context = rag_search_tool_func(topic)
        prompt = f"Generate 5 multiple-choice questions with correct answers based on this course material:\n{context}\n\nFormat: Question | A) | B) | C) | D) | Answer: X"
        return llm.predict(prompt)

    def summary_tool_func(text: str) -> str:
        prompt = f"Summarize this educational content concisely, keeping key concepts:\n{text}"
        return llm.predict(prompt)

    tools = [
        Tool(name="RAG_Search", func=rag_search_tool_func,
             description="Retrieve relevant content from uploaded course PDFs. Returns exact excerpts with source citations."),
        Tool(name="Generate_Notes", func=generate_notes_tool_func,
             description="Generate structured, hierarchical study notes for a given topic from course materials."),
        Tool(name="Generate_Quiz", func=generate_quiz_tool_func,
             description="Generate 5 multiple-choice practice questions with answers based on course content."),
        Tool(name="Summarize", func=summary_tool_func,
             description="Summarize complex educational content into digestible key points.")
    ]

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        agent_kwargs={
            "system_message": SYSTEM_PROMPT
        }
    )
    return agent