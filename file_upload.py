import google.generativeai as genai
import streamlit as st
import os

def upload_files_to_gemini(api_key, uploaded_files):
    """Upload files to Gemini"""
    genai.configure(api_key=api_key)
    file_ids = []
    file_paths = []
    
    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            file_path = os.path.join(os.getcwd(), uploaded_file.name)
            
            # Save uploaded file temporarily
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            # Upload to Gemini
            response = genai.upload_file(
                path=file_path,
                mime_type='application/pdf'
            )
            
            file_ids.append(response.name)
            file_paths.append(file_path)
            print(f"✓ Uploaded: {uploaded_file.name}")
    
    return file_ids, file_paths

def check_and_upload_files(api_key):
    """Check and upload files"""
    genai.configure(api_key=api_key)
    
    st.warning("Upload Educational Material (PDF)")
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

    if st.button("Upload Files"):
        if uploaded_files:
            try:
                file_ids, file_paths = upload_files_to_gemini(api_key, uploaded_files)
                st.success(f"{len(file_ids)} files successfully uploaded.")
                return file_ids, file_paths
            except Exception as e:
                st.error(f"An error occurred: {e}")
                return [], []
        else:
            st.warning("Please select at least one file.")
            return [], []
    
    return [], []