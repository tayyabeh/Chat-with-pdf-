import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_community.embeddings import FastEmbedEmbeddings
from groq import Groq
from langchain.chains import LLMChain
from langchain_core.language_models import LLM
from langchain_groq import ChatGroq

# Initialize Groq client
llm = ChatGroq(
    api_key="your api key here",
    model_name="mixtral-8x7b-32768"
)

def initialize_embeddings():
    """Set up our document embedding model for searching through PDF content"""
    embeddings = FastEmbedEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        cache_folder='./.cache'
    )
    return embeddings

def get_pdf_text(pdf_docs):
    """Extract all text content from uploaded PDF files"""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Break down the PDF text into manageable chunks for processing"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    chunks = splitter.split_text(text)
    return chunks

def get_vector_store(chunks):
    """Create a searchable database from our text chunks"""
    try:
        with st.spinner("Processing documents..."):
            embeddings = initialize_embeddings()
            vector_store = FAISS.from_texts(
                texts=chunks,
                embedding=embeddings
            )
            vector_store.save_local("faiss_index")
            return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        raise

def get_conversational_chain():
    """Create the QA chain for processing questions"""
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}\n
    Question: \n{question}\n

    Answer:
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    chain = LLMChain(
        llm=llm,
        prompt=prompt
    )
    return chain

def user_input(user_question):
    """Process user questions using the vector store"""
    try:
        embeddings = initialize_embeddings()
        vector_store = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True  # Enable dangerous deserialization
        )
        
        # Get relevant documents
        docs = vector_store.similarity_search(user_question)
        
        # Extract text from docs for context
        context = " ".join([doc.page_content for doc in docs])
        
        chain = get_conversational_chain()
        response = chain.run(
            context=context,
            question=user_question
        )
        
        return response
        
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")
        return None

def clear_chat_history():
    """Reset the chat history"""
    st.session_state.messages = [
        {"role": "assistant", "content": "Upload some PDFs and ask me a question"}
    ]

def main():
    # Set up the Streamlit page
    st.set_page_config(
        page_title="PDF Chatbot",
        page_icon="🤖"
    )

    # Create the sidebar for PDF uploads
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button",
            accept_multiple_files=True
        )
        
        if st.button("Submit & Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file.")
                return
                
            with st.spinner("Processing PDFs..."):
                try:
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Processing completed successfully!")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

    # Set up the main chat interface
    st.title("Chat with PDF files 🤖")
    st.write("Welcome to the chat!")
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    # Initialize or load chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Upload some PDFs and ask me a question"}
        ]

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Handle new user input
    if prompt := st.chat_input():
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Generate and display assistant response
        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = user_input(prompt)
                    if response:
                        st.write(response)
                        message = {"role": "assistant", "content": response}
                        st.session_state.messages.append(message)

if __name__ == "__main__":
    main()