from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from ocr_preprocessing import OCRPreprocessor
import os
import time
import logging
from observability import tracker

logger = logging.getLogger(__name__)

RELEVANCE_THRESHOLD = 0.3
ocr_preprocessor = OCRPreprocessor()  # Initialize once

def setup_rag_pipeline(pdf_paths, api_key):
    """Initialize RAG with vector search from PDFs (with OCR for scanned PDFs)"""
    
    all_documents = []
    
    for pdf_path in pdf_paths:
        try:
            if os.path.exists(pdf_path):
                print(f"Loading: {pdf_path}")
                
                # Detect if scanned and use appropriate loader
                if ocr_preprocessor.is_scanned_pdf(pdf_path):
                    print(f"  → Scanned PDF detected, using OCR...")
                    documents = ocr_preprocessor.hybrid_load_pdf(pdf_path)
                else:
                    print(f"  → Native text PDF, using standard extraction...")
                    from pypdf import PdfReader
                    from langchain.schema import Document
                    reader = PdfReader(pdf_path)
                    documents = []
                    for page_num, page in enumerate(reader.pages, 1):
                        text = page.extract_text() or ""
                        metadata = {
                            "source": os.path.basename(pdf_path),
                            "page": page_num,
                            "ocr_preprocessed": False
                        }
                        documents.append(Document(page_content=text, metadata=metadata))
                
                all_documents.extend(documents)
                print(f"✓ Loaded {len(documents)} pages from {pdf_path}")
            else:
                print(f"✗ File not found: {pdf_path}")
        except Exception as e:
            logger.error(f"Error loading {pdf_path}: {e}")
            print(f"Error loading {pdf_path}: {e}")
    
    if not all_documents:
        print("✗ No documents loaded")
        return None
    
    print(f"Total documents loaded: {len(all_documents)}")
    
    # Split documents into chunks (preserve paragraphs for formulas)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(all_documents)
    print(f"Created {len(chunks)} chunks")
    
    # Create embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
    
    # Create vector store
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("✓ Vector store created")
    
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key,
        temperature=0.2
    )
    
    # Create RAG chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 6}),
        return_source_documents=True
    )
    
    return qa_chain

def compact_context(documents: list, max_length: int = 2000) -> str:
    """Context Compaction: Summarize long documents to reduce token load"""
    if not documents:
        return ""
    
    # Concatenate all document content
    full_text = "\n\n".join([doc.page_content for doc in documents])
    
    # Truncate if too long (token reduction)
    if len(full_text) > max_length:
        full_text = full_text[:max_length] + "..."
    
    return full_text

def query_rag(qa_chain, question):
    """Query the RAG pipeline with semantic filtering and context compaction"""
    retrieval_start = time.time()
    
    result = qa_chain({"query": question})
    
    retrieval_time = time.time() - retrieval_start
    
    response = result["result"]
    sources = result.get("source_documents", [])
    
    # Log retrieval metrics
    tracker.log_retrieval(question, retrieval_time, len(sources))
    
    # Estimate token count (rough: 1 token ≈ 4 chars)
    total_chars = len(question) + len(response)
    token_estimate = total_chars // 4
    tracker.log_token_count(question, response, token_estimate)
    
    # Apply context compaction to reduce token usage
    if sources:
        compacted_context = compact_context(sources, max_length=2000)
    
    # Format source references with relevance info
    source_info = "\n\n**📚 Sources Used:**\n"
    for i, doc in enumerate(sources, 1):
        filename = os.path.basename(doc.metadata.get('source', 'Unknown'))
        page = doc.metadata.get('page', 'N/A')
        source_info += f"{i}. **{filename}** (Page {page})\n"
    
    return response + source_info