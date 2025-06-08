import streamlit as st
import os
import nltk
import tempfile
import shutil

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import NLTKTextSplitter
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Constants
CHROMA_DB_PATH = "./chroma_db_streamlit"   # Directory to store ChromaDB for the Streamlit app

@st.cache_resource(show_spinner="Checking NLTK tokenizers...")
def download_nltk_punkt():
    nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
    if not os.path.exists(nltk_data_path):
        os.makedirs(nltk_data_path, exist_ok=True)
    
    if nltk_data_path not in nltk.data.path:
        nltk.data.path.append(nltk_data_path)

    resources_to_check = ['punkt', 'punkt_tab']
    all_resources_found = True # Assume all found initially

    for resource in resources_to_check:
        try:
            nltk.data.find(f'tokenizers/{resource}', paths=[nltk_data_path])
            st.sidebar.info(f"NLTK '{resource}' resource found locally.")
        except LookupError:
            st.sidebar.warning(f"NLTK '{resource}' resource not found locally. Attempting download...")
            try:
                nltk.download(resource, download_dir=nltk_data_path, quiet=False)
                # Verify download
                nltk.data.find(f'tokenizers/{resource}', paths=[nltk_data_path])
                st.sidebar.success(f"NLTK '{resource}' resource downloaded to {nltk_data_path}.")
            except Exception as e:
                st.sidebar.error(f"Failed to download NLTK '{resource}' resource: {e}")
                st.error(
                    f"Critical Error: Could not download NLTK '{resource}' resource. "
                    f"Details: {e}\n\n"
                    "The application may not function correctly without this resource. "
                    "Please ensure you have a stable internet connection or try installing it manually."
                )
                all_resources_found = False # Mark that at least one resource failed
    
    if not all_resources_found:
        st.warning("One or more NLTK resources could not be downloaded or verified. Please check the messages above.")
        # Optionally, st.stop() here if all resources are absolutely critical for the app to even start
    elif resources_to_check: # Only show success if we actually checked for resources
        st.sidebar.success("All required NLTK tokenizers are available.")

def format_docs(docs):
    """Helper function to format retrieved documents for the prompt."""
    return "\n\n".join(doc.page_content for doc in docs)

@st.cache_resource(show_spinner="Initializing AI Models...")
def get_ai_models(api_key):
    """Initializes and returns the chat and embedding models."""
    try:
        chat_model = ChatGoogleGenerativeAI(google_api_key=api_key,
                                          model="gemini-2.0-flash-exp")
        embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=api_key,
                                                     model="models/embedding-001")
        return chat_model, embedding_model
    except Exception as e:
        st.error(f"Failed to initialize Google AI models: {e}\n" +
                 "Please ensure your API key is correct and has access to the specified models.")
        return None, None

@st.cache_resource(show_spinner="Processing PDF and creating/loading vector store...")
def get_retriever(_embedding_model, uploaded_file, db_path):
    """Loads uploaded PDF, processes it, and creates/loads a Chroma vector store retriever."""
    
    if uploaded_file is None:
        st.info("Please upload a PDF file to begin.")
        return None

    temp_dir = tempfile.mkdtemp()
    temp_pdf_path = os.path.join(temp_dir, uploaded_file.name)

    try:
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.info(f"Processing '{uploaded_file.name}'...")

        if os.path.exists(db_path):
            st.sidebar.info(f"Clearing existing vector store at '{db_path}' for new file.")
            try:
                shutil.rmtree(db_path)
            except Exception as e:
                st.error(f"Error clearing existing vector store: {e}")
        
        st.info(f"Creating new vector store at '{db_path}'. This may take a moment...")
        loader = PyPDFLoader(temp_pdf_path)
        pages = loader.load_and_split()

        if not pages:
            st.error("No content found in the PDF.")
            return None

        text_splitter = NLTKTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(pages)

        if not chunks:
            st.error("No text chunks were extracted from the PDF. Cannot proceed.")
            return None
        
        db = Chroma.from_documents(chunks, _embedding_model, persist_directory=db_path)
        st.success(f"Vector store for '{uploaded_file.name}' created and persisted at '{db_path}'.")
        
        db_connection = Chroma(persist_directory=db_path, embedding_function=_embedding_model)
        return db_connection.as_retriever(search_kwargs={"k": 5})

    except Exception as e:
        st.error(f"Error during PDF processing or vector store operation: {e}")
        return None
    finally:
        if os.path.exists(temp_dir):
             shutil.rmtree(temp_dir)

def main():
    st.set_page_config(page_title="LangChain RAG with Gemini", layout="wide")
    download_nltk_punkt()
    st.title("📄 LangChain RAG System with Google Gemini")

    st.sidebar.header("🔑 Configuration")
    google_api_key = st.sidebar.text_input("Enter your Google API Key:", type="password")
    
    st.sidebar.header("📁 Upload PDF")
    uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

    if not google_api_key:
        st.warning("Please enter your Google API Key in the sidebar to proceed.")
        st.stop()

    chat_model, embedding_model = get_ai_models(google_api_key)

    if not chat_model or not embedding_model:
        st.error("AI Models could not be initialized. Please check your API key and model access.")
        st.stop()
    
    st.sidebar.success("AI Models Initialized.")

    retriever = None
    current_doc_name = "No document processed yet."

    if uploaded_file is not None:
        # Call get_retriever only if a file is uploaded
        retriever = get_retriever(embedding_model, uploaded_file, CHROMA_DB_PATH)
        if retriever:
            current_doc_name = uploaded_file.name
            st.success(f"Retriever is ready for '{current_doc_name}'. You can now ask questions.")
        else:
            st.error(f"Failed to initialize document retriever for '{uploaded_file.name}'. Check logs for details.")
            current_doc_name = f"Failed to process {uploaded_file.name}."
    else:
        st.info("Please upload a PDF document in the sidebar to enable question answering.")

    # Define RAG chain and UI for asking questions only if retriever is ready
    if retriever:
        chat_template = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a Helpful AI Bot.
                        Given a context and question from user,
                        you should answer based on the given context."""),
            HumanMessagePromptTemplate.from_template("""Answer the question based on the given context.
            Context: {context}
            Question: {question}
            Answer: """)
        ])
        output_parser = StrOutputParser()
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | chat_template
            | chat_model
            | output_parser
        )

        st.header(f"💬 Ask a Question about '{current_doc_name}'")
        user_question = st.text_input("Enter your question:")

        if user_question:
            with st.spinner("🧠 Thinking..."):
                try:
                    response = rag_chain.invoke(user_question)
                    st.subheader("💡 Answer:")
                    st.markdown(response)
                except Exception as e:
                    st.error(f"Error generating answer: {e}")
    else:
        st.header("💬 Ask a Question")
        st.markdown("Please upload a PDF and ensure the retriever is initialized to ask questions.")

if __name__ == "__main__":
    main()
