import os
import tempfile
import streamlit as st
from typing import List, Any, Tuple, Dict, Optional

# Import required libraries
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda

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

# New function for Self-RAG
def should_retrieve(llm, query: str, chat_history: List) -> bool:
    """Determines if retrieval is necessary based on the query and chat history."""
    
    if not chat_history:
        # Always retrieve for the first question
        return True
    
    # Create the decision prompt
    system_prompt = """You are an assistant that decides whether a new user question requires retrieving information from a document database.
    
    You should answer YES if:
    - The user is asking a new question about document content
    - The user is asking for specific information not covered in previous responses
    - The user is asking about a new topic or entity not previously discussed
    
    You should answer NO if:
    - The user is asking for clarification about your previous response
    - The user is asking a follow-up that can be answered with information already provided
    - The user is making small talk or asking a general question not related to documents
    - The user is asking you to explain or elaborate on something you've already said
    
    Reply with ONLY "YES" or "NO".
    """
    
    # Format the chat history for the prompt
    formatted_history = ""
    for i, message in enumerate(chat_history[-6:]):  # Only use the last 6 messages to keep context manageable
        role = "Human" if i % 2 == 0 else "Assistant"
        formatted_history += f"{role}: {message}\n"
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Chat history:\n{formatted_history}\n\nNew user question: {query}\n\nShould I retrieve information from the documents to answer this question? Answer YES or NO.")
    ]
    
    # Get the decision
    response = llm.invoke(messages).content.strip().upper()
    return "YES" in response

def create_memory() -> Any:
    return ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

# New answer generation function for Self-RAG
def generate_answer(llm, query: str, chat_history: List, retrieved_docs=None) -> str:
    """Generates an answer with or without retrieved documents."""
    
    if retrieved_docs:
        # If we have retrieved documents, use them
        system_prompt = """You are a helpful assistant answering questions about documents. 
        Base your answer on the provided document excerpts when relevant. 
        If the document excerpts don't contain the answer, you can use your general knowledge but clearly indicate when you are doing so.
        Provide detailed, informative responses."""
        
        # Format the retrieved documents
        docs_text = "\n\n".join([f"Document excerpt {i+1}:\n{doc.page_content}" for i, doc in enumerate(retrieved_docs)])
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Document excerpts:\n{docs_text}\n\nQuestion: {query}")
        ]
    else:
        # If we don't have retrieved documents, use conversation context
        system_prompt = """You are a helpful assistant engaged in a conversation.
        Answer based on the conversation history and your general knowledge.
        Provide detailed, informative responses."""
        
        formatted_history = ""
        for i, message in enumerate(chat_history[-10:]):  # Use the last 10 messages
            role = "Human" if i % 2 == 0 else "Assistant"
            formatted_history += f"{role}: {message}\n"
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Conversation history:\n{formatted_history}\n\nQuestion: {query}")
        ]
    
    # Generate the answer
    return llm.invoke(messages).content.strip()

# Modified query processing for Self-RAG
def process_query_self_rag(chain: Any, llm: Any, retriever: Any, query: str, chat_history: List) -> Tuple[str, List[Any]]:
    """Process a query using the Self-RAG approach."""
    
    # Step 1: Decide whether to retrieve
    retrieval_needed = should_retrieve(llm, query, chat_history)
    
    # Step 2: Retrieve if necessary
    retrieved_docs = None
    if retrieval_needed:
        retrieved_docs = retriever.get_relevant_documents(query)
    
    # Step 3: Generate answer
    answer = generate_answer(llm, query, chat_history, retrieved_docs)
    
    # Return the answer and the retrieved documents (if any)
    return answer, retrieved_docs if retrieval_needed else []

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
    
    # Step 5: Create LLM and traditional chain (as fallback)
    progress_text.text("Finalizing setup...")
    llm = create_llm(groq_api_key, groq_model)
    memory = create_memory()
    traditional_chain = create_rag_chain(retriever, llm, memory)
    progress_bar.progress(1.0)
    
    # Clear progress indicators
    progress_text.empty()
    
    # Return both the traditional chain and the components for self-RAG
    return {
        "traditional_chain": traditional_chain,
        "llm": llm,
        "retriever": retriever
    }

def create_rag_chain(retriever: Any, llm: Any, memory: Any) -> Any:
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
    )

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
        page_title="Simple Chat with Self-RAG",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None

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
                        rag_system = setup_rag_system(
                            file_paths=file_paths,
                            groq_api_key=groq_api_key,
                        )
                        
                        st.session_state.rag_system = rag_system
                        st.success(f"Successfully processed {len(file_paths)} documents!")
                    except Exception as e:
                        st.error(f"Error setting up RAG system: {str(e)}")

        if st.session_state.file_paths:
            st.subheader("Processed Documents:")
            for file_path in st.session_state.file_paths:
                st.write(f"- {os.path.basename(file_path)}")
                
        # Add a toggle for Self-RAG
        st.subheader("Advanced Settings")
        use_self_rag = st.checkbox("Use Self-RAG", value=True, help="Enable intelligent retrieval decisions")
        if use_self_rag:
            st.session_state.use_self_rag = True
        else:
            st.session_state.use_self_rag = False

    st.title("Simple Chat with Self-RAG")
    
    if st.session_state.use_self_rag:
        st.caption("🧠 Self-RAG enabled: Intelligently deciding when to retrieve document information")

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
        st.session_state.chat_history.append(prompt)
        
        with st.chat_message("user"):
            st.write(prompt)
        
        if st.session_state.rag_system is None:
            with st.chat_message("assistant"):
                st.write("Please upload documents and process them first.")
                st.session_state.messages.append({"role": "assistant", "content": "Please upload documents and process them first."})
        else:
            with st.chat_message("assistant"):
                with st.spinner("Generating response..."):
                    try:
                        # Use traditional chain or Self-RAG based on setting
                        if st.session_state.use_self_rag:
                            # Get components
                            llm = st.session_state.rag_system["llm"]
                            retriever = st.session_state.rag_system["retriever"]
                            
                            # Process with Self-RAG
                            response, sources = process_query_self_rag(
                                None, 
                                llm, 
                                retriever, 
                                prompt, 
                                st.session_state.chat_history
                            )
                        else:
                            # Use traditional chain
                            traditional_chain = st.session_state.rag_system["traditional_chain"]
                            response, sources = traditional_chain.invoke({"question": prompt})["answer"], traditional_chain.invoke({"question": prompt})["source_documents"]
                        
                        st.write(response)
                        
                        # Display sources if available
                        if sources:
                            retrieval_status = "📚 Retrieved from documents"
                            with st.expander("Sources"):
                                for i, source in enumerate(sources):
                                    st.write(f"**Source {i+1}:** {source.metadata.get('source', 'Unknown')}, Page: {source.metadata.get('page', 'N/A')}")
                                    st.write(f"Content: {source.page_content[:500]}...")
                        else:
                            retrieval_status = "💬 Answered from conversation context"
                            
                        # Show retrieval status if using Self-RAG
                        if st.session_state.use_self_rag:
                            st.caption(retrieval_status)
                        
                        # Add to session states
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response,
                            "sources": sources
                        })
                        st.session_state.chat_history.append(response)
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": f"Error: {str(e)}"
                        })

    if not st.session_state.rag_system:
        st.info("👈 Please upload and process your documents in the sidebar to start chatting")

if __name__ == "__main__":
    main()