import streamlit as st
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
    Answer the question as detailed as possible from the provided context, and keep chat history in mind and make sure to provide all the details. Don't give wrong information; give answer from context only. If the question is not related to context, simply respond 'out of context'. Kindly go through the chat history also.

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

def main():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello! I'm a document assistant. Ask me anything about the documents"),
        ]

    st.set_page_config(page_title="Chat PDF")
    
    st.header("Carnot Research")
    st.markdown("Chat with documents")
    st.subheader("Upload your Documents")

    pdf_docs = st.file_uploader("Upload your PDF, DOC, DOCX, or TXT Files and Click on the Submit & Process Button", accept_multiple_files=True)

    if st.button("Submit & Process"):
        if pdf_docs:
            valid_files = True
            for doc in pdf_docs:
                if not (doc.name.endswith(".pdf") or doc.name.endswith(".doc") or doc.name.endswith(".docx") or doc.name.endswith(".txt")):
                    valid_files = False
                    break

            if valid_files:
                with st.spinner("Processing..."):
                    raw_text = get_text_from_files(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")
            else:
                st.error("Please upload files in PDF, DOC, DOCX, or TXT format.")
        else:
            st.error("No files uploaded. Please upload one or more documents.")

    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)

    user_query = st.chat_input("Type a message...")

    if user_query and user_query.strip() != "":
        st.session_state.chat_history.append(HumanMessage(content=user_query))

        with st.chat_message("Human"):
            st.markdown(user_query)

        with st.chat_message("AI"):
            response = user_input(user_query)
            res = response["output_text"]
            st.markdown(res)
            st.session_state.chat_history.append(AIMessage(content=res))

if __name__ == "__main__":
    try:
        main()
    except Exception as e :
        st.markdown(e)

