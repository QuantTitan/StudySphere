import google.generativeai as genai
from rag_engine import setup_rag_pipeline, query_rag
import os

# Cache RAG pipeline
rag_chain = None
cached_file_paths = None

def send_query_get_response(api_key, user_question, file_paths):
    """Send query using RAG pipeline"""
    global rag_chain, cached_file_paths
    
    # Verify files exist
    valid_paths = [p for p in file_paths if os.path.exists(p)]
    
    if not valid_paths:
        print("⚠ No valid file paths found")
        return direct_gemini_response(api_key, user_question)
    
    # Reinitialize if files changed
    if rag_chain is None or cached_file_paths != valid_paths:
        print(f"Initializing RAG pipeline with {len(valid_paths)} files...")
        rag_chain = setup_rag_pipeline(valid_paths, api_key)
        cached_file_paths = valid_paths
        
        if rag_chain is None:
            return "Error: Could not load documents for RAG."
    
    # Query RAG
    try:
        response = query_rag(rag_chain, user_question)
        print(f"✓ RAG Response generated from {len(valid_paths)} files")
        return response
    except Exception as e:
        print(f"RAG Error: {e}")
        return direct_gemini_response(api_key, user_question)

def direct_gemini_response(api_key, user_question):
    """Fallback: Direct response without RAG"""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    message = model.generate_content([user_question], stream=False)
    return message.text