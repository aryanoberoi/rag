from flask import Flask, request, jsonify, render_template, session
from flask_session import Session
import asyncio
from PyPDF2 import PdfReader
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
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Ensure the environment variables are loaded
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

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
        f.write(docx_file.read())
    return docx2txt.process(temp_file_path)

def get_text_from_txt(txt_file):
    return txt_file.read().decode("utf-8")

def get_text_from_pdf(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_from_files(files):
    text = ""
    for file in files:
        if file.filename.endswith(".pdf"):
            text += get_text_from_pdf(file)
        elif file.filename.endswith(".doc"):
            text += get_text_from_doc(file)
        elif file.filename.endswith(".docx"):
            text += get_text_from_docx(file)
        elif file.filename.endswith(".txt"):
            text += get_text_from_txt(file)
        else:
            return f"Unsupported file type: {file.filename}"
    return text

def get_text_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        text = ' '.join(p.get_text() for p in soup.find_all('p'))
        return text
    except Exception as e:
        return f"Error fetching the URL: {e}"

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks
# take foldername input 
def get_vector_store(text_chunks, usersession):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(usersession)

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

def user_input(user_question, usersession):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local(usersession, embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}, StrOutputParser()
    )
    return response

@app.route('/', methods=['GET', 'POST'])
def index():
    message = None  # Initialize a message variable
    file_details = []  # Initialize a list to store file details
    url_displayed = session.get('url_input', '')  # Retrieve the stored URL or set it to empty string

    if 'session_id' not in session:
        session['session_id'] = os.urandom(24).hex()
        session['chat_history'] = [
            AIMessage(content="Hello! I'm a document assistant. Ask me anything about the documents you upload."),
        ]

    if request.method == 'POST':
        files = request.files.getlist("files")
        url_input = request.form.get("url_input")
        raw_text = ""

        if files and files[0].filename != '':
            valid_files = all(f.filename.endswith(('.pdf', '.doc', '.docx', '.txt')) for f in files)
            if valid_files:
                raw_text = get_text_from_files(files)
                message = "Files successfully uploaded."

                # Get file details for display
                for file in files:
                    file_details.append({"name": file.filename})
            else:
                message = "Please upload files in PDF, DOC, DOCX, or TXT format."
        elif url_input:
            raw_text = get_text_from_url(url_input)
            message = "URL processed successfully. URl: " + url_input
            session['url_input'] = url_input  # Store the URL in the session
        else:
            message = "No files uploaded or URL provided. Please upload documents or provide a URL."

        if raw_text:
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks, session['session_id'])

    chat_history = session.get('chat_history', [])
    return render_template('index.html', chat_history=chat_history, message=message, file_details=file_details, url_displayed=url_displayed)

@app.route('/ask', methods=['POST'])
def ask():
    user_query = request.json.get("question")
    if user_query and user_query.strip():
        session['chat_history'].append(HumanMessage(content=user_query))
        response = user_input(user_query, session['session_id'])
        res = response["output_text"]
        session['chat_history'].append(AIMessage(content=res))
        return jsonify({"answer": res, "url": session.get('url_input', '')})
    return jsonify({"error": "Invalid input"})


if __name__ == "__main__":
    app.run(debug=True)
