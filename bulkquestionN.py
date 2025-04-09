import os
import json
from dotenv import load_dotenv
import streamlit as st
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import pandas as pd
import time

# Load environment variables
load_dotenv()
os.environ['NVIDIA_API_KEY'] = os.getenv("NVIDIA_API_KEY")

# Directories for vector database and documents
VECTORDIR = r"F:\OsBook\unix\common\Vcetor"
DOCUMENT_DIR = r"F:\OsBook\unix\common"

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
        embeddings = NVIDIAEmbeddings()
        st.session_state.vectors = FAISS.load_local(VECTORDIR, embeddings, allow_dangerous_deserialization=True)
        st.success("Vector database loaded successfully.")
    except Exception as e:
        st.error(f"Error loading vector database: {e}")

def vector_embedding():
    """Generates embeddings and stores them in a vector database."""
    if "vectors" not in st.session_state:
        st.session_state.embeddings = NVIDIAEmbeddings()
        text_loader = DirectoryLoader(DOCUMENT_DIR, glob="**/*.txt", loader_cls=TextLoader)
        pdf_loader = PyPDFDirectoryLoader(DOCUMENT_DIR)
        docs = text_loader.load() + pdf_loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
        final_documents = text_splitter.split_documents(docs)
        st.session_state.vectors = FAISS.from_documents(final_documents, st.session_state.embeddings)
        st.success("Vector Store DB is ready.")

def process_bulk_questions(questions_json):
    """Processes multiple questions from a JSON input."""
    try:
        questions = json.loads(questions_json)
        if not isinstance(questions, list):
            st.error("Invalid JSON format. Please provide a list of questions.")
            return None

        responses_with_references = []
        retriever = st.session_state.vectors.as_retriever()
        llm = ChatNVIDIA(model="meta/llama3-70b-instruct")
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        for item in questions:
            if "question" in item:
                question = item["question"]
                response = retrieval_chain.invoke({"input": question})
                responses_with_references.append({"question": question, "answer": response["answer"]})
            else:
                st.warning(f"Skipping item due to missing 'question' key: {item}")
        return responses_with_references
    except json.JSONDecodeError:
        st.error("Invalid JSON format. Please ensure your JSON is well-formed.")
        return None

def create_pdf_report(responses, filename="unix_ans.pdf"):
    """Generates a PDF report with questions, answers, and references."""
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    flowables = []

    for response in responses:
        question_text = Paragraph(f"<b>Question:</b> {response['question']}", styles['Normal'])
        answer_text = Paragraph(f"<b>Answer:</b> {response['answer']}", styles['Normal'])
        flowables.append(question_text)
        flowables.append(answer_text)
        flowables.append(Spacer(1, 12))

    doc.build(flowables)
    st.success(f"PDF report '{filename}' generated successfully!")

# Prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context first. If the answer is not found in the context, provide the most accurate response based on your understanding.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

# Streamlit Interface
st.title("AskNLearn")

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Save Vector DB"):
        save_vector_db()
with col2:
    if st.button("Load Vector DB"):
        load_vector_db()
with col3:
    if st.button("Generate Embeddings"):
        vector_embedding()

try:
    df = pd.read_csv("questions.csv")  # Change filename if needed
    questions = [{"question": q} for q in df["question"].tolist()]
except FileNotFoundError:
    st.warning("questions.csv not found. Using default questions.")
    questions = []

bulk_input_text = st.text_area(
    "Enter multiple questions in JSON format:",
    value=json.dumps(questions, indent=2),
    height=200,
)

if st.button("Get Answers in PDF"):
    if "vectors" in st.session_state:
        responses = process_bulk_questions(bulk_input_text)
        if responses:
            create_pdf_report(responses)
            with open("unix_ans.pdf", "rb") as file:
                st.download_button(
                    label="Download PDF Report",
                    data=file,
                    file_name="unix_ans.pdf",
                    mime="application/pdf",
                )
        else:
            st.warning("No valid questions found in JSON input.")
    else:
        st.error("Vector database is not loaded. Please generate or load the vector database first.")

st.markdown("---")
input_text = st.text_input("Or ask a single question:")

if input_text:
    if "vectors" in st.session_state:
        llm = ChatNVIDIA(model="meta/llama3-70b-instruct")
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        start_time = time.process_time()
        response = retrieval_chain.invoke({"input": input_text})
        st.write(response["answer"])
        st.write(f"Response time: {time.process_time() - start_time} seconds")

        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")
    else:
        st.error("Vector database is not loaded. Please generate or load the vector database first.")
