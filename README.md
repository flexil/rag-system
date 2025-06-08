# LangChain RAG System with Google Gemini & Streamlit

A Retrieval Augmented Generation (RAG) application that allows users to upload PDF documents and ask questions about their content. The system leverages Google's Gemini AI models for language understanding and generation, with LangChain orchestrating the RAG pipeline and Streamlit providing an interactive web interface.

## Features

*   **PDF Upload**: Users can upload their own PDF files for processing.
*   **Dynamic RAG Pipeline**: For each uploaded PDF, the system:
    *   Loads and parses the PDF content.
    *   Splits the text into manageable chunks using NLTK.
    *   Generates embeddings for each chunk using Google's embedding model.
    *   Stores these embeddings in a Chroma vector database for efficient retrieval.
*   **Question Answering**: Users can ask natural language questions about the content of the uploaded PDF.
*   **Google Gemini Integration**: Utilizes Google Gemini Pro for answering questions based on the retrieved context and Google's embedding model for document embedding.
*   **Interactive UI**: Built with Streamlit for a user-friendly experience, including API key input, file upload, and display of results.
*   **NLTK Tokenizer Management**: Automatically checks for and downloads necessary NLTK tokenizers (`punkt` and `punkt_tab`) to a local project directory (`nltk_data`).
*   **Caching**: Employs Streamlit's caching for AI model initialization and vector store loading/creation to improve performance on subsequent runs with the same PDF (though currently, the vector store is cleared for each new PDF upload for simplicity).

## Prerequisites

*   Python 3.8 or higher
*   Access to Google AI Studio and a Google API Key with the Gemini API enabled.

## Installation

1.  **Clone the repository (if applicable) or download the project files.**
    ```bash
    # If you have it in a git repository:
    # git clone <repository-url>
    # cd LangChain-RAG-System-with-Google-Gemini-main
    ```

2.  **Create and activate a Python virtual environment:**
    This is highly recommended to manage dependencies and avoid conflicts with system-wide packages.
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
    On Windows, activation is `venv\Scripts\activate`.

3.  **Install the required Python packages:**
    Ensure your virtual environment is active, then run:
    ```bash
    pip install -r requirements.txt
    ```
    The `requirements.txt` file should include:
    ```
    streamlit
    langchain
    langchain-google-genai
    langchain-community
    langchain-text-splitters
    langchain-chroma
    pypdf
    nltk
    chromadb
    ```

## Obtaining and Using Your Google API Key

1.  Go to [Google AI Studio](https://aistudio.google.com/).
2.  Sign in with your Google account.
3.  Create a new API key or use an existing one. Ensure the Gemini API is enabled for this key.
4.  When you run the Streamlit application, you will be prompted to enter this API key in the sidebar. The application requires this key to interact with Google's AI models.

## Running the Application

1.  **Ensure your virtual environment is activated:**
    ```bash
    source venv/bin/activate
    ```

2.  **Navigate to the project directory** (e.g., `LangChain-RAG-System-with-Google-Gemini-main`).

3.  **Run the Streamlit application:**
    ```bash
    streamlit run streamlit_app.py
    ```
    Alternatively, if you encounter issues with the `streamlit` command not being found directly, you can use:
    ```bash
    python -m streamlit run streamlit_app.py
    ```
    (Ensure you use `venv/bin/python` if you need to be explicit about the interpreter from your virtual environment).

4.  The application should open in your default web browser. If not, the terminal will display local and network URLs (e.g., `http://localhost:8501`).

5.  **Using the App:**
    *   Enter your Google API Key in the sidebar.
    *   Upload a PDF file using the file uploader in the sidebar.
    *   Once the PDF is processed and the retriever is ready, you can type your question into the main input box and press Enter.
    *   The answer, based on the PDF's content, will be displayed.

## Project Structure

```
. (root directory)
├── streamlit_app.py       # Main Streamlit application script
├── requirements.txt       # Python package dependencies
├── nltk_data/             # (Auto-created) Stores NLTK tokenizers
├── chroma_db_streamlit/   # (Auto-created) Stores Chroma vector database for the current PDF
├── LeaveNoContextBehind.pdf # (Example PDF, if included initially)
└── README.md              # This file
```

## Key Technologies Used

*   **Streamlit**: For building the interactive web application UI.
*   **LangChain**: As the framework for orchestrating the RAG pipeline, including document loading, text splitting, embedding, vector storage, retrieval, and LLM interaction.
*   **Google Gemini**: Specifically `gemini-2.0-flash-exp` (or similar) for chat/question-answering and `models/embedding-001` for text embeddings.
*   **PyPDFLoader (LangChain Community)**: For loading text content from PDF files.
*   **NLTKTextSplitter (LangChain Text Splitters)**: For splitting documents into manageable chunks, using NLTK's sentence tokenization capabilities.
*   **Chroma**: As the vector store for storing and retrieving document embeddings.
*   **NLTK**: For natural language processing tasks, specifically sentence tokenization required by `NLTKTextSplitter`.

---
