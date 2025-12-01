import google.generativeai as genai
import streamlit as st
import os
from pathlib import Path

def upload_files_to_gemini(api_key, uploaded_files):
    """Upload files to Gemini and save locally"""
    genai.configure(api_key=api_key)
    
    # Create temp directory for PDFs
    temp_dir = "temp_pdfs"
    os.makedirs(temp_dir, exist_ok=True)
    
    file_ids = []
    file_paths = []
    
    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            # Save file locally with proper path
            file_path = os.path.join(temp_dir, uploaded_file.name)
            
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            file_paths.append(file_path)
            print(f"✓ Saved locally: {file_path}")
            
            try:
                # Upload to Gemini (optional, for backup)
                response = genai.upload_file(
                    path=file_path,
                    mime_type='application/pdf'
                )
                file_ids.append(response.name)
                print(f"✓ Uploaded to Gemini: {uploaded_file.name}")
            except Exception as e:
                print(f"⚠ Gemini upload failed: {e}")
    
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
                st.success(f"{len(file_paths)} files successfully uploaded and saved.")
                st.session_state.file_paths = file_paths  # Store in session
                return file_ids, file_paths
            except Exception as e:
                st.error(f"An error occurred: {e}")
                return [], []
        else:
            st.warning("Please select at least one file.")
            return [], []
    
    # Return previously saved files if they exist
    if "file_paths" in st.session_state:
        return [], st.session_state.file_paths
    
    return [], []