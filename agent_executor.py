from langchain.agents import initialize_agent, AgentType, Tool
from llm_client import get_llm
from agent_tools import (
    rag_search_tool,
    generate_notes_tool,
    generate_quiz_tool,
    summary_tool
)

tools = [
    Tool(
        name="RAG_Search",
        func=rag_search_tool,
        description="Use this tool to retrieve content from uploaded PDFs."
    ),
    Tool(
        name="Generate_Notes",
        func=generate_notes_tool,
        description="Generate structured notes for any given topic."
    ),
    Tool(
        name="Generate_Quiz",
        func=generate_quiz_tool,
        description="Generate 5 MCQ quiz questions for a topic."
    ),
    Tool(
        name="Summarize",
        func=summary_tool,
        description="Summarize any provided text."
    )
]

# Main Agent
agent = initialize_agent(
    tools,
    get_llm(),
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)
