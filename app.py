import streamlit as st
from langchain_community.chat_models import ChatOllama
# Add this code before import ib_insync:
#ERROR
import asyncio

def get_or_create_eventloop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return asyncio.get_event_loop()

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.messages import AIMessage,HumanMessage
from langchain_core.output_parsers import StrOutputParser


# load_dotenv()
# os.getenv("GOOGLE_API_KEY")
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
#improve
    prompt_template = """
    Answer the question as detailed as possible from the provided context,and keep chat history in mind and make sure to provide all the details, dont give wrong information give answer from context only if that question is not related to context simply respond out of context ,but kindly go through the chat history also\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro")
    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , StrOutputParser(),)
    
    return response



def main():
    if "chat_history" not in st.session_state:
    #we can initialize it empty but i will initialize it with a AI Message
        st.session_state.chat_history=[
            AIMessage(content="Hello! I'm a PDF assistant. Ask me anything about the documents"),
        ]
    st.set_page_config("Chat PDF",initial_sidebar_state="expanded")
    st.title("Carnot Research")
    st.title("Chat with PDF")
    st.subheader("Upload Your Documents")
    pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
    if st.button("Submit & Process"):
        with st.spinner("Processing..."):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.success("Done")

    for message in st.session_state.chat_history:
        if isinstance(message,AIMessage):
            with st.chat_message("AI"):
                st.markdown(message.content)
        elif isinstance(message,HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)     
            
#for storing the messages of AI and Human in history of user question
    user_query=st.chat_input("Type a message...")

    if user_query is not None and user_query.strip!="":
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        
        #displaying the message in the conver
        with st.chat_message("Human"):
            st.markdown(user_query)
        
        with st.chat_message("AI"):
            # if is_valid(user_query,SQLDatabase)=="NOT_RELATED":
            #     model = genai.GenerativeModel("gemini-pro")
            #     response=model.generate_content([user_query])
            #     st.markdown(response.text)
            #     st.session_state.chat_history.append(AIMessage(content=response.text))
            # else:   
            response=user_input(user_query)
            print(response)
            res=response["output_text"]
            st.write(res)
            st.session_state.chat_history.append(AIMessage(content=res))

        


if __name__ == "__main__":
    main()
