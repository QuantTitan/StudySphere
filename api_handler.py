import google.generativeai as genai
import time

def send_query_get_response(api_key, user_question, file_paths):
    """Send query to Gemini with file context"""
    genai.configure(api_key=api_key)
    
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    # Add file reference instruction
    user_question = user_question + ' and tell me which file are the top results based on your similarity search.'
    
    # Create message with files
    message = model.generate_content(
        [user_question],
        stream=False
    )
    
    response = message.text
    print(f"✓ Response generated: {response[:100]}...")
    
    return response if response else "Server issue, try again"


