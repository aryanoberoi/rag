# Chat PDF using Gemini

This project is a Streamlit application that allows users to chat with their PDF documents using Google's Gemini model for conversational AI. Users can upload Documents or Links, and the application processes these  to enable natural language queries about their content. The responses are generated using advanced AI embeddings and conversational chains provided by Google Generative AI.

## Features

- **DocumentUpload**: Users can upload multiple format documents for processing.
- **URL Upload**: Users can upload URL or document.
- **Text Extraction**: Extracts text from the uploaded files or URL.
- **Text Chunking**: Splits the extracted text into manageable chunks for efficient processing.
- **Vector Store**: Creates and saves a vector store using FAISS for fast similarity searches.
- **Conversational AI**: Utilizes Google's Gemini model to answer user queries based on the content of the uploaded PDFs.
- **Session Management**: Maintains chat history within the session.

## Requirements

- Python 3.7 or higher
- Streamlit
- PyPDF2
- Langchain
- FAISS
- Google Generative AI
- Python dotenv

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/chat-pdf-gemini.git
    cd chat-pdf-gemini
    ```

2. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Set up your Google API Key**:
    - Create a `.env` file in the root directory.
    - Add your Google API Key to the `.env` file:
      ```env
      GOOGLE_API_KEY=your_google_api_key
      ```

## Usage

1. **Run the Streamlit application**:
    ```bash
    streamlit run app.py
    ```

2. **Upload PDFs**:
    - Use the sidebar to upload your PDF files.
    - Click on the "Submit & Process" button to process the uploaded PDFs.

3. **Chat with your PDFs**:
    - Type your questions in the chat input field.
    - The AI will respond based on the content of the uploaded PDFs.

## Code Overview

### `app.py`

This is the main file of the Streamlit application.

- **Imports**: Imports necessary libraries and modules.
- **Event Loop Management**: Ensures compatibility with `asyncio` in a Streamlit environment.
- **PDF Processing Functions**:
  - `get_pdf_text`: Extracts text from uploaded PDF documents.
  - `get_text_chunks`: Splits the extracted text into chunks.
  - `get_vector_store`: Creates and saves a FAISS vector store from text chunks.
- **Conversational Chain**:
  - `get_conversational_chain`: Sets up the conversational chain using a custom prompt template.
- **User Input Handling**:
  - `user_input`: Handles user queries and retrieves responses from the conversational AI model.
- **Main Function**:
  - `main`: Manages the Streamlit application layout and user interaction.

### `.env`

This file contains your Google API key.

### `requirements.txt`

This file lists all the Python dependencies required for the project.


## Acknowledgements

- [Streamlit](https://www.streamlit.io/)
- [PyPDF2](https://pypdf2.readthedocs.io/en/latest/)
- [Langchain](https://www.langchain.com/)
- [FAISS](https://faiss.ai/)
- [Google Generative AI](https://ai.google/)

---

