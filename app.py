import streamlit as st
from PIL import Image
from api_handler import send_query_get_response
from chat_gen import generate_html
from file_upload import check_and_upload_files
import os

logo = Image.open('logo.png')
sb_logo = Image.open('sb_logo.png')

c1, c2 = st.columns([0.9, 3.2])

with c1:
    st.caption('')
    st.caption('')
    st.image(logo, width=120)

with c2:
    st.title('EduMentor : An AI-Enhanced Tutoring System')

st.markdown("## AI Tutor Description")
rag_description = """
EduMentor leverages RAG (Retrieval-Augmented Generation) with Google Gemini to provide in-depth, contextually rich answers to complex educational queries. It retrieves relevant information from your uploaded documents and generates accurate responses.
"""
st.markdown(rag_description)

# Gemini API Key Input
api_key = st.text_input(label='Enter your Gemini API Key', type='password')

if api_key:
    file_ids, file_paths = check_and_upload_files(api_key)
    
    st.markdown(f'Number of files uploaded: :blue[{len(file_ids)}]')
    st.divider()

    # Sidebar
    st.sidebar.header('EduMentor: AI-Tutor')
    st.sidebar.image(logo, width=120)
    st.sidebar.caption('Made by D')

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
                response = send_query_get_response(api_key, prompt, file_paths)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

else:
    st.warning("Please enter your Gemini API Key to use EduMentor.")