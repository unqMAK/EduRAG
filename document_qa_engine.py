import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader, TextLoader, UnstructuredPowerPointLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain

load_dotenv()

# Set up environment variables
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
# os.environ["LANGCHAIN_TRACING_V2"] = "false"
# os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# Directories for vector database and documents
VECTORDIR = r"F:\OsBook\rtos"

# Define prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions:{input}
    """
)

# Initialize Gemma model
llm = Ollama(model="gemma2:2b")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# Streamlit UI
def save_vector_db():
    """Saves the FAISS vector database to the specified directory."""
    if "vectors" in st.session_state:
        try:
            st.session_state.vectors.save_local(VECTORDIR)
            st.success("Vector database saved successfully.")
        except Exception as e:
            st.error(f"Error saving vector database: {e}")
    else:
        st.error("No vector database found. Please generate embeddings first.")

def load_vector_db():
    """Loads the FAISS vector database from the specified directory."""
    try:
        st.session_state.vectors = FAISS.load_local(VECTORDIR, OllamaEmbeddings(model='gemma2:2b'),allow_dangerous_deserialization=True)
        st.success("Vector database loaded successfully.")
    except Exception as e:
        st.error(f"Error loading vector database: {e}")

def vector_embedding():
    """Generates embeddings and stores them in a vector database."""
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings(model="gemma2:2b")
        st.session_state.loader = DirectoryLoader(r"F:\OsBook", glob="**/*.txt", loader_cls=TextLoader)
        st.session_state.loader = PyPDFDirectoryLoader(r"F:\OsBook")
        st.session_state.docs=st.session_state.loader.load()
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=700,chunk_overlap=50) 
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:30]) 
        print("hEllo")
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings) 


        # Initialize the embedding model
       

        # Generate the FAISS vector database
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

st.title("Ask & Learn with Gemma")


# Buttons for vector DB management
if st.button("Save Vector Database"):
    save_vector_db()

if st.button("Load Vector Database"):
    load_vector_db()

if st.button("Generate Document Embeddings"):
    vector_embedding()
    st.write("Vector Store DB is ready.")

# User input for question
input_text = st.text_input("What question do you have in mind?")

if input_text:
    if "vectors" in st.session_state:
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, chain)
        response = retrieval_chain.invoke({"input": input_text})
        st.write(response["answer"])

        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")
    else:
        st.error("Vector database is not loaded. Please generate or load the vector database first.")
