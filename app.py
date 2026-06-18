import os
import streamlit as st
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter          # ✅ still correct
from langchain_community.embeddings import HuggingFaceEmbeddings            # ✅ use community
from langchain_community.vectorstores import Chroma                         # ✅ use community
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import tempfile
import re
# ------------------------------
# 1. SETUP: API Key (from environment)
# ------------------------------
# Make sure GOOGLE_API_KEY is set in your environment
# If not, you can set it here:
os.environ["GOOGLE_API_KEY"] = ""

# ------------------------------
# 2. PDF EXTRACTION (local, no cloud)
# ------------------------------
def extract_text_from_pdf(pdf_path):
    """Extract text and tables from a PDF using pdfplumber."""
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Extract text
            text = page.extract_text()
            if text:
                full_text += f"\n--- Page {page.page_number} ---\n{text}\n"
            
            # Extract tables and convert to text
            tables = page.extract_tables()
            for table in tables:
                if table and len(table) > 0:
                    table_str = "\n".join([" | ".join([str(cell) if cell else "" for cell in row]) for row in table if any(row)])
                    full_text += f"\n[Table on page {page.page_number}]:\n{table_str}\n"
    return full_text

# ------------------------------
# 3. CHUNKING
# ------------------------------
def chunk_documents(text, source_name):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n--- Page", "\n\n", "\n", ".", " "],
    )
    # We create a dummy document with metadata
    docs = text_splitter.create_documents(
        texts=[text],
        metadatas=[{"source": source_name}]
    )
    # Add page numbers from the text itself (we inserted them)
    for doc in docs:
        # Extract page number from chunk if present
        content = doc.page_content
        if "--- Page " in content:
            # crude extraction
            import re
            match = re.search(r"--- Page (\d+) ---", content)
            if match:
                doc.metadata["page"] = int(match.group(1))
    return docs

# ------------------------------
# 4. VECTOR STORE (local Chroma)
# ------------------------------
def build_or_load_vectorstore(docs, persist_dir="./faoreports_db"):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        # Load existing
        vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    else:
        # Create new
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=persist_dir
        )
        vectorstore.persist()
    return vectorstore

# ------------------------------
# 5. STREAMLIT APP
# ------------------------------
def main():
    st.set_page_config(page_title="FAO Reports Chatbot", page_icon="📘")
    st.title("📘 FAO Reports Q&A (with Gemini 2.5 Flash)")

    # Initialise session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    # Sidebar: Upload PDFs
    with st.sidebar:
        st.header("📄 Upload PDFs")
        uploaded_files = st.file_uploader(
            "Upload one or more FAO reports (PDF)",
            type=["pdf"],
            accept_multiple_files=True
        )
        
        if st.button("💾 Process PDFs"):
            if uploaded_files:
                all_docs = []
                with st.spinner("Extracting and indexing PDFs..."):
                    for uploaded_file in uploaded_files:
                        # Save temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            tmp_file.write(uploaded_file.read())
                            tmp_path = tmp_file.name
                        
                        # Extract text
                        text = extract_text_from_pdf(tmp_path)
                        # Chunk
                        docs = chunk_documents(text, uploaded_file.name)
                        all_docs.extend(docs)
                        
                        # Clean up temp file
                        os.unlink(tmp_path)
                    
                    # Build vector store
                    vectorstore = build_or_load_vectorstore(all_docs)
                    st.session_state.vectorstore = vectorstore
                    
                    # Build QA chain
                    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
                    memory = ConversationBufferMemory(
                        memory_key="chat_history",
                        return_messages=True,
                        output_key="answer"
                    )
                    
                    # Gemini LLM
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-2.5-flash",
                        temperature=0.2,
                        convert_system_message_to_human=True,
                    )
                    
                    # Custom prompt with citations
                    prompt_template = """You are a helpful assistant answering questions about FAO reports. 
                    Answer ONLY based on the provided context. For every factual statement, cite the source page like [Page X].

                    Context: {context}

                    Chat history: {chat_history}

                    Question: {question}

                    Answer:"""
                    PROMPT = PromptTemplate(
                        template=prompt_template,
                        input_variables=["context", "chat_history", "question"]
                    )
                    
                    qa_chain = ConversationalRetrievalChain.from_llm(
                        llm=llm,
                        retriever=retriever,
                        memory=memory,
                        return_source_documents=True,
                        combine_docs_chain_kwargs={"prompt": PROMPT},
                        verbose=False,
                    )
                    
                    st.session_state.qa_chain = qa_chain
                    st.success(f"✅ Processed {len(uploaded_files)} PDFs! You can now ask questions.")
            else:
                st.warning("Please upload at least one PDF.")

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about the reports..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if st.session_state.qa_chain is None:
                    st.warning("Please upload and process PDFs first.")
                else:
                    try:
                        response = st.session_state.qa_chain({"question": prompt})
                        answer = response["answer"]
                        sources = response.get("source_documents", [])
                        
                        # Display answer
                        st.markdown(answer)
                        
                        # Display sources
                        if sources:
                            with st.expander("📚 Sources"):
                                for doc in sources:
                                    source = doc.metadata.get("source", "Unknown")
                                    page = doc.metadata.get("page", "N/A")
                                    st.write(f"- **{source}**, Page {page}")
                        else:
                            st.caption("No source documents retrieved.")
                        
                        # Save assistant message
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        st.error(f"Error: {e}")

if __name__ == "__main__":
    main()