from langchain_google_genai import ChatGoogleGenerativeAI

def get_llm(api_key: str):
    """Return a Gemini model client using LangChain and the provided api_key."""
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key,
        temperature=0.7
    )
