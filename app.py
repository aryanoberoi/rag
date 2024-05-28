import streamlit as st
import asyncio
from PyPDF2 import PdfReader
from werkzeug.utils import secure_filename
from docx import Document
import docx2txt
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores.faiss import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
import requests
from bs4 import BeautifulSoup
from flask import Flask, request, render_template, redirect, url_for, session, flash
# Ensure the environment variables are loaded
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = Flask(__name__)
app.secret_key = os.urandom(24)
# Function to handle asyncio event loop
def get_or_create_eventloop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return asyncio.get_event_loop()
get_or_create_eventloop()

def get_text_from_doc(doc_file):
    document = Document(doc_file)
    return "\n".join([paragraph.text for paragraph in document.paragraphs])

def get_text_from_docx(docx_file):
    # Save the uploaded file to a temporary location to be processed by docx2txt
    temp_file_path = "temp.docx"
    with open(temp_file_path, "wb") as f:
        f.write(docx_file.getbuffer())
    return docx2txt.process(temp_file_path)

def get_text_from_txt(txt_file):
    return txt_file.getvalue().decode("utf-8")

def get_text_from_pdf(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_from_files(files):
    text = ""
    for file in files:
        if file.name.endswith(".pdf"):
            text += get_text_from_pdf(file)
        elif file.name.endswith(".doc"):
            text += get_text_from_doc(file)
        elif file.name.endswith(".docx"):
            text += get_text_from_docx(file)
        elif file.name.endswith(".txt"):
            text += get_text_from_txt(file)
        else:
            st.error(f"Unsupported file type: {file.name}")
    return text

def get_text_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        text = ' '.join(p.get_text() for p in soup.find_all('p'))
        return text
    except Exception as e:
        st.error(f"Error fetching the URL: {e}")
        return ""

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
   Answer the question as detailed as possible. Please ensure that the answer is detailed and accurate based on the provided context. Review the chat history carefully to provide all necessary details and avoid incorrect information. Treat synonyms or similar words as equivalent within the context. For example, if a question refers to "modules" or "units" instead of "chapters" or "doc" instead of "document" consider them the same. If the question is not related to the provided context, simply respond with "out of context."


    Context:\n{context}?\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro")
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}, StrOutputParser()
    )
    return response

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'document' in request.files:
            file = request.files['document']
            if file.filename != '':
                filename = secure_filename(file.filename)
                file_path = os.path.join('/path/to/upload/folder', filename)
                file.save(file_path)
                # Process the file based on its type
                raw_text = get_text_from_files([file])  # Adjust this function to work with Flask file storage
                # Process text, etc.
                return redirect(url_for('chat'))
        elif 'url' in request.form:
            url = request.form['url']
            if url:
                # Process the URL
                raw_text = get_text_from_url(url)
                # Process text, etc.
                return redirect(url_for('chat'))
    return render_template('home.html')

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        user_query = request.form['message']
        if user_query:
            # Handle chat interaction here
            response = user_input(user_query)  # Define user_input to integrate with your chat model
            flash(response)
            return redirect(url_for('chat'))
    return render_template('chat.html')



if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)


