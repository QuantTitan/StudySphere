import google.generativeai as genai
from rag_engine import setup_rag_pipeline, query_rag

# Cache RAG pipeline
rag_chain = None

def send_query_get_response(api_key, user_question, file_paths):
    """Send query using RAG pipeline"""
    global rag_chain
    
    if not file_paths:
        # Fallback to direct Gemini if no files
        return direct_gemini_response(api_key, user_question)
    
    # Initialize RAG pipeline if not already done
    if rag_chain is None:
        print("Initializing RAG pipeline...")
        rag_chain = setup_rag_pipeline(file_paths, api_key)
    
    if rag_chain is None:
        return "Error: Could not load documents for RAG."
    
    # Query RAG
    response = query_rag(rag_chain, user_question)
    print(f"✓ RAG Response generated")
    
    return response if response else "Server issue, try again"

def direct_gemini_response(api_key, user_question):
    """Fallback: Direct response without RAG"""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    message = model.generate_content([user_question], stream=False)
    return message.text