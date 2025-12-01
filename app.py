import streamlit as st
from PIL import Image
from api_handler import send_query_get_response
from chat_gen import generate_html
from file_upload import check_and_upload_files
import os
from agent_executor import create_agent
from rag_engine import setup_rag_pipeline

logo = Image.open('logo.png')
sb_logo = Image.open('sb_logo.png')

c1, c2 = st.columns([0.9, 3.2])

with c1:
    st.caption('')
    st.caption('')
    st.image(logo, width=120)

with c2:
    st.title('StudySphere : An AI-Enhanced Tutoring System')

st.markdown("## AI Tutor Description")
rag_description = """
StudySphere leverages RAG (Retrieval-Augmented Generation) with Google Gemini to provide in-depth, contextually rich answers to complex educational queries. It retrieves relevant information from your uploaded documents and generates accurate responses.

**✨ Advanced Features:**
- 🔄 **Context Compaction** - Reduces token load while preserving key information
- 🎯 **Semantic Filtering** - Ensures high-relevance results only (>0.7 similarity)
- 🧠 **Role-Enriched Prompts** - Expert educational guidance with source citations
"""
st.markdown(rag_description)

# Gemini API Key Input
api_key = st.text_input(label='Enter your Gemini API Key', type='password')

if api_key:
    # create agent for this api_key if not present or if key changed
    if "agent_api_key" not in st.session_state or st.session_state.agent_api_key != api_key:
        st.session_state.agent = create_agent(api_key)
        st.session_state.agent_api_key = api_key

    file_ids, file_paths = check_and_upload_files(api_key)

    if file_paths:
        # initialize RAG chain using the same api_key
        if "qa_chain" not in st.session_state:
            st.session_state.qa_chain = setup_rag_pipeline(file_paths, api_key)
    
    st.markdown(f'Number of files uploaded: :blue[{len(file_ids)}]')
    st.divider()

    # Sidebar
    st.sidebar.header('StudySphere: AI-Tutor')
    st.sidebar.image(logo, width=120)
    st.sidebar.caption('Made by QuantTitan and D')
    
    # Show active enhancements
    st.sidebar.markdown("### 🚀 Active Enhancements")
    st.sidebar.markdown("✓ Context Compaction\n✓ Semantic Filtering\n✓ Role-Enriched Prompts")

    if st.sidebar.button('Generate Chat History'):
        if "messages" in st.session_state:
            html_data = generate_html(st.session_state.messages)
            st.sidebar.download_button(
                label="Download Chat History as HTML",
                data=html_data,
                file_name="chat_history.html",
                mime="text/html"
            )

    # Main Chat Interface
    st.subheader('Q&A record with AI-Tutor 📜')
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Welcome and ask a question to the AI tutor"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                # use session agent created with user's api_key
                response = st.session_state.agent.run(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

else:
    st.warning("Please enter your Gemini API Key to use StudySphere.")