# # llm_client.py
# from vertexai.generative_models import GenerativeModel

# def get_llm():
#     """Return the Gemini model instance."""
#     return GenerativeModel("gemini-1.5-flash")

# llm_client.py

from langchain_google_genai import ChatGoogleGenerativeAI

def get_llm():
    """Return a Gemini model client using LangChain."""
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key='',
        temperature=0.7
    )
