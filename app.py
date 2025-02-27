import os
import sys
import tempfile
import streamlit as st
from typing import List, Any, Tuple

# Import required libraries
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

def load_document(file_path: str) -> List[Any]:
    _, file_extension = os.path.splitext(file_path)
    
    if file_extension.lower() == '.pdf':
        loader = PyPDFLoader(file_path)
    elif file_extension.lower() == '.txt':
        loader = TextLoader(file_path)
    elif file_extension.lower() in ['.docx', '.doc']:
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")
    
    return loader.load()

def load_documents(file_paths: List[str]) -> List[Any]:
    documents = []
    for file_path in file_paths:
        documents.extend(load_document(file_path))
    return documents

def chunk_documents(documents: List[Any], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Any]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    return text_splitter.split_documents(documents)

def create_vector_store(chunks: List[Any], embeddings: Any) -> Any:
    vectordb = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    
    return vectordb

def create_llm(api_key: str, model_name: str = "llama3-8b-8192") -> Any:
    return ChatGroq(
        api_key=api_key,
        model_name=model_name,
        temperature=0.2,
        max_tokens=4096
    )

def create_memory() -> Any:
    return ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

def create_rag_chain(retriever: Any, llm: Any, memory: Any) -> Any:
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
    )

def process_query(chain: Any, query: str) -> Tuple[str, List[Any]]:
    result = chain.invoke({"question": query})
    return result['answer'], result['source_documents']

def setup_rag_system(
    file_paths: List[str],
    groq_api_key: str,
) -> Any:
    # Fixed parameters
    chunk_size = 1000
    chunk_overlap = 200
    groq_model = "llama3-8b-8192"
    
    # Add progress placeholder in the UI
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    # Step 1: Load documents
    progress_text.text("Loading documents...")
    documents = load_documents(file_paths)
    progress_bar.progress(0.2)
    
    # Step 2: Chunk documents
    progress_text.text("Chunking text...")
    chunks = chunk_documents(documents, chunk_size, chunk_overlap)
    progress_bar.progress(0.4)
    
    # Step 3: Create embeddings
    progress_text.text("Creating embeddings...")
    embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )
    progress_bar.progress(0.6)
    
    # Step 4: Create vector store
    progress_text.text("Building vector store...")
    vectordb = create_vector_store(chunks, embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    progress_bar.progress(0.8)
    
    # Step 5: Create chain
    progress_text.text("Finalizing setup...")
    llm = create_llm(groq_api_key, groq_model)
    memory = create_memory()
    chain = create_rag_chain(retriever, llm, memory)
    progress_bar.progress(1.0)
    
    # Clear progress indicators
    progress_text.empty()
    
    return chain

def save_uploaded_files(uploaded_files) -> List[str]:
    file_paths = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(st.session_state.temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_paths.append(file_path)
    return file_paths

def main():
    st.set_page_config(
        page_title="Simple Chat",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "chain" not in st.session_state:
        st.session_state.chain = None

    if "file_paths" not in st.session_state:
        st.session_state.file_paths = []

    if "temp_dir" not in st.session_state:
        st.session_state.temp_dir = tempfile.mkdtemp()

    with st.sidebar:
        st.title("Document Upload")
        
        # Get API key from Streamlit secrets or UI input
        groq_api_key = ""
        if 'GROQ_API_KEY' in st.secrets:
            groq_api_key = st.secrets['GROQ_API_KEY']
            st.success("API key loaded from secrets!")
        else:
            groq_api_key = st.text_input("Enter your Groq API Key:", type="password")
            if groq_api_key:
                os.environ["GROQ_API_KEY"] = groq_api_key
        
        uploaded_files = st.file_uploader(
            "Upload your documents:", 
            accept_multiple_files=True,
            type=["pdf", "txt", "docx"]
        )
        
        if st.button("Process Documents"):
            if not groq_api_key:
                st.error("Please enter your Groq API Key")
            elif not uploaded_files:
                st.error("Please upload at least one document")
            else:
                with st.spinner("Processing documents..."):
                    file_paths = save_uploaded_files(uploaded_files)
                    st.session_state.file_paths = file_paths
                    
                    try:
                        chain = setup_rag_system(
                            file_paths=file_paths,
                            groq_api_key=groq_api_key,
                        )
                        
                        st.session_state.chain = chain
                        st.success(f"Successfully processed {len(file_paths)} documents!")
                    except Exception as e:
                        st.error(f"Error setting up RAG system: {str(e)}")

        if st.session_state.file_paths:
            st.subheader("Processed Documents:")
            for file_path in st.session_state.file_paths:
                st.write(f"- {os.path.basename(file_path)}")

    st.title("Simple Chat")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("Sources"):
                    for i, source in enumerate(message["sources"]):
                        st.write(f"**Source {i+1}:** {source.metadata.get('source', 'Unknown')}, Page: {source.metadata.get('page', 'N/A')}")
                        st.write(f"Content: {source.page_content[:500]}...")

    if prompt := st.chat_input("Ask a question about your documents"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        if st.session_state.chain is None:
            with st.chat_message("assistant"):
                st.write("Please upload documents and process them first.")
                st.session_state.messages.append({"role": "assistant", "content": "Please upload documents and process them first."})
        else:
            with st.chat_message("assistant"):
                with st.spinner("Generating response..."):
                    try:
                        response, sources = process_query(st.session_state.chain, prompt)
                        
                        st.write(response)
                        if sources:
                            with st.expander("Sources"):
                                for i, source in enumerate(sources):
                                    st.write(f"**Source {i+1}:** {source.metadata.get('source', 'Unknown')}, Page: {source.metadata.get('page', 'N/A')}")
                                    st.write(f"Content: {source.page_content[:500]}...")
                        
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response,
                            "sources": sources
                        })
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": f"Error: {str(e)}"
                        })

    if not st.session_state.chain:
        st.info("👈 Please upload and process your documents in the sidebar to start chatting")

if __name__ == "__main__":
    main()