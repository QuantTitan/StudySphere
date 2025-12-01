import streamlit as st
from PIL import Image
from api_handler import send_query_get_response
from chat_gen import generate_html
from file_upload import check_and_upload_files
import os
from agent_executor import (
    create_rag_agent,
    create_tutor_agent,
    create_notes_agent,
    create_summarizer_agent,
)
from rag_engine import setup_rag_pipeline
from observability import tracker
from agent_evaluator import AgentEvaluator, TEST_PROMPTS

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
- 🎯 **Semantic Filtering** - Ensures high-relevance results only (>0.3 similarity)
- 🧠 **Role-Enriched Prompts** - Expert educational guidance with source citations
"""
st.markdown(rag_description)

# Gemini API Key Input
api_key = st.text_input(label='Enter your Gemini API Key', type='password')

if api_key:
    # create agents for this api_key if not present or if key changed
    if "agent_api_key" not in st.session_state or st.session_state.agent_api_key != api_key:
        st.session_state.rag_agent = create_rag_agent(api_key)
        st.session_state.tutor_agent = create_tutor_agent(api_key)
        st.session_state.notes_agent = create_notes_agent(api_key)
        st.session_state.summarizer_agent = create_summarizer_agent(api_key)
        st.session_state.agent_api_key = api_key

    file_ids, file_paths = check_and_upload_files(api_key)

    if file_paths:
        # initialize RAG chain using the same api_key
        if "qa_chain" not in st.session_state:
            st.session_state.qa_chain = setup_rag_pipeline(file_paths, api_key)
    
    st.markdown(f'Number of files uploaded: :blue[{len(file_ids)}]')
    st.divider()

    # Sidebar: pick agent role
    agent_choice = st.sidebar.radio("Choose agent role", ["Tutor (Q&A)", "RAG Retrieval", "Notes Generator", "Summarizer"])
    
    # Sidebar: Observability & Evaluation
    st.sidebar.divider()
    st.sidebar.subheader("📊 Observability")
    
    if st.sidebar.button("View Metrics"):
        metrics = tracker.get_summary()
        st.sidebar.json(metrics)
    
    if st.sidebar.button("Export Metrics"):
        tracker.export_metrics()
        st.sidebar.success("Metrics exported to metrics.json")
    
    st.sidebar.divider()
    st.sidebar.subheader("🔍 Agent Evaluation")
    
    if st.sidebar.button("Run Test Suite"):
        evaluator = AgentEvaluator(api_key)
        st.session_state.evaluator = evaluator
        st.sidebar.info("Test suite initialized. Use 'Evaluate Response' to assess agent.")
    
    if st.sidebar.button("Evaluate Last Response"):
        if "evaluator" in st.session_state and st.session_state.messages:
            evaluator = st.session_state.evaluator
            last_user = None
            last_response = None
            
            # Find last Q&A pair
            for i in range(len(st.session_state.messages) - 1, -1, -1):
                if st.session_state.messages[i]["role"] == "assistant":
                    last_response = st.session_state.messages[i]["content"]
                elif st.session_state.messages[i]["role"] == "user":
                    last_user = st.session_state.messages[i]["content"]
                    break
            
            if last_user and last_response:
                eval_result = evaluator.evaluate_response(last_user, last_response, "Course materials")
                st.sidebar.json(eval_result)
    
    if st.sidebar.button("Export Evaluations"):
        if "evaluator" in st.session_state:
            st.session_state.evaluator.export_evaluations()
            st.sidebar.success("Evaluations exported to agent_evaluations.json")
    
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
                # route to selected agent and enforce RAG precondition
                if agent_choice in ("RAG Retrieval", "Notes Generator", "Tutor (Q&A)") and ("qa_chain" not in st.session_state or st.session_state.qa_chain is None):
                    response = "⚠️ Please upload documents first to enable RAG-powered agents."
                else:
                    import time
                    start = time.time()
                    if agent_choice == "RAG Retrieval":
                        response = st.session_state.rag_agent.run(prompt)
                    elif agent_choice == "Notes Generator":
                        response = st.session_state.notes_agent.run(prompt)
                    elif agent_choice == "Summarizer":
                        response = st.session_state.summarizer_agent.run(prompt)
                    else:
                        response = st.session_state.tutor_agent.run(prompt)
                    elapsed = time.time() - start
                    tracker.log_query(prompt, agent_choice, response, elapsed)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

else:
    st.warning("Please enter your Gemini API Key to use StudySphere.")