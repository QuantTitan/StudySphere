import google.generativeai as genai
import streamlit as st
import os
from ocr_preprocessing import OCRPreprocessor

def upload_files_to_gemini(api_key, uploaded_files):
    """Upload files to Gemini and save locally"""
    genai.configure(api_key=api_key)
    
    temp_dir = "temp_pdfs"
    os.makedirs(temp_dir, exist_ok=True)
    
    file_ids = []
    file_paths = []
    
    ocr_preprocessor = OCRPreprocessor()
    
    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            file_paths.append(file_path)
            
            # Check if scanned and display status
            is_scanned = ocr_preprocessor.is_scanned_pdf(file_path)
            status = "🖼️ Scanned (OCR)" if is_scanned else "📄 Native Text"
            print(f"✓ Saved locally: {file_path} [{status}]")
            
            try:
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
    """Check and upload files with OCR detection"""
    genai.configure(api_key=api_key)
    
    st.warning("Upload Educational Material (PDF)")
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

    if st.button("Upload Files"):
        if uploaded_files:
            try:
                file_ids, file_paths = upload_files_to_gemini(api_key, uploaded_files)
                st.success(f"{len(file_paths)} files successfully uploaded and saved.")
                
                # Show OCR status for each file
                ocr_preprocessor = OCRPreprocessor()
                for fp in file_paths:
                    is_scanned = ocr_preprocessor.is_scanned_pdf(fp)
                    status = "🖼️ Scanned (OCR will be used)" if is_scanned else "📄 Native Text"
                    st.info(f"{os.path.basename(fp)}: {status}")
                
                st.session_state.file_paths = file_paths
                return file_ids, file_paths
            except Exception as e:
                st.error(f"An error occurred: {e}")
                return [], []
        else:
            st.warning("Please select at least one file.")
            return [], []
    
    if "file_paths" in st.session_state:
        return [], st.session_state.file_paths
    
    return [], []