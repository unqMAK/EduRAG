import os
import json
from dotenv import load_dotenv
import streamlit as st
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader, TextLoader, UnstructuredPowerPointLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain, RetrievalQA
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import pandas as pd
import time

# Load environment variables
load_dotenv()
os.environ['NVIDIA_API_KEY'] = os.getenv("NVIDIA_API_KEY")

# Directories for vector databases and documents
NVIDIA_VECTORDIR = r"F:\OsBook\unix\common\NVIDIA_Vector"
OLLAMA_VECTORDIR = r"F:\OsBook\QB\OLLAMA_Vector"
DOCUMENT_DIR = r"F:\OsBook\unix\common"
OLLAMA_DOC_DIR = r"C:\Users\pjite\Desktop\RTOS\QB" # Adjust this path

# --- NVIDIA Model Setup ---
nvidia_prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context first. If the answer is not found in the context, provide the most accurate response based on your understanding.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

# --- Ollama Model Setup ---
ollama_qa_prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context only.
    Please provide the most accurate and concise response based on the question.
    Cite the source document and page number where you found the information.
    You can process the context to your understanding and answer the question.
    <context>
    {context}
    <context>
    Question:{input}
    """
)

ollama_question_generation_prompt = ChatPromptTemplate.from_template(
    """
    Generate a question based on the following text snippet. The question should be clear, concise, and relevant to the content.
    Text snippet: {text}
    """
)
ollama_llm = Ollama(model="gemma2:2b")
ollama_output_parser = st.empty() # Will be used later
ollama_qa_chain = ollama_qa_prompt | ollama_llm | st.empty() # Will be used later
ollama_question_generation_chain = ollama_question_generation_prompt | ollama_llm | st.empty() # Will be used later

# --- Helper Functions ---
def save_vector_db(vectordir, vectors):
    """Saves the FAISS vector database to the specified directory."""
    if vectors:
        try:
            vectors.save_local(vectordir)
            st.success(f"Vector database saved successfully to {vectordir}.")
        except Exception as e:
            st.error(f"Error saving vector database: {e}")
    else:
        st.error("No vector database found. Please generate embeddings first.")

def load_vector_db(vectordir, embeddings):
    """Loads the FAISS vector database from the specified directory."""
    try:
        vectors = FAISS.load_local(vectordir, embeddings, allow_dangerous_deserialization=True)
        st.success(f"Vector database loaded successfully from {vectordir}.")
        return vectors
    except Exception as e:
        st.error(f"Error loading vector database: {e}")
        return None

def create_pdf_report(responses, filename):
    """Generates a PDF report with questions and answers."""
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

# --- Streamlit UI ---
st.title("Multi-Model Question Answering Agent")

model_option = st.radio("Select Model:", ["NVIDIA", "Ollama (local)"])

if model_option == "NVIDIA":
    # --- NVIDIA Specific UI ---
    st.subheader("NVIDIA Model Operations")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Save NVIDIA Vector DB"):
            save_vector_db(NVIDIA_VECTORDIR, st.session_state.get("nvidia_vectors"))
    with col2:
        if st.button("Load NVIDIA Vector DB"):
            st.session_state.nvidia_embeddings = NVIDIAEmbeddings()
            st.session_state.nvidia_vectors = load_vector_db(NVIDIA_VECTORDIR, st.session_state.nvidia_embeddings)
    with col3:
        if st.button("Generate NVIDIA Embeddings"):
            with st.spinner("Generating NVIDIA Embeddings..."):
                st.session_state.nvidia_embeddings = NVIDIAEmbeddings()
                text_loader = DirectoryLoader(DOCUMENT_DIR, glob="**/*.txt", loader_cls=TextLoader)
                pdf_loader = PyPDFDirectoryLoader(DOCUMENT_DIR)
                docs = text_loader.load() + pdf_loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
                final_documents = text_splitter.split_documents(docs)
                st.session_state.nvidia_vectors = FAISS.from_documents(final_documents, st.session_state.nvidia_embeddings)
                st.success("NVIDIA Vector Store DB is ready.")

    bulk_input_text = st.text_area(
        "Enter multiple questions in JSON format for NVIDIA:",
        value=json.dumps([{"question": "What is a process?"}, {"question": "Explain virtual memory."}], indent=2),
        height=200,
    )

    def process_nvidia_bulk_questions(questions_json):
        """Processes multiple questions for NVIDIA model from a JSON input."""
        try:
            questions = json.loads(questions_json)
            if not isinstance(questions, list):
                st.error("Invalid JSON format. Please provide a list of questions.")
                return None

            responses_with_references = []
            if "nvidia_vectors" not in st.session_state:
                st.error("NVIDIA Vector database is not loaded.")
                return None
            retriever = st.session_state.nvidia_vectors.as_retriever()
            llm = ChatNVIDIA(model="meta/llama3-70b-instruct")
            document_chain = create_stuff_documents_chain(llm, nvidia_prompt)
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

    if st.button("Get NVIDIA Answers in PDF"):
        if "nvidia_vectors" in st.session_state:
            responses = process_nvidia_bulk_questions(bulk_input_text)
            if responses:
                create_pdf_report(responses, "nvidia_answers.pdf")
                with open("nvidia_answers.pdf", "rb") as file:
                    st.download_button(
                        label="Download NVIDIA PDF Report",
                        data=file,
                        file_name="nvidia_answers.pdf",
                        mime="application/pdf",
                    )
            else:
                st.warning("No valid questions found in JSON input for NVIDIA.")
        else:
            st.error("NVIDIA Vector database is not loaded. Please generate or load the vector database first.")

    st.markdown("---")
    nvidia_input_text = st.text_input("Or ask a single question to NVIDIA:")

    if nvidia_input_text:
        if "nvidia_vectors" in st.session_state:
            llm = ChatNVIDIA(model="meta/llama3-70b-instruct")
            document_chain = create_stuff_documents_chain(llm, nvidia_prompt)
            retriever = st.session_state.nvidia_vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            start_time = time.process_time()
            response = retrieval_chain.invoke({"input": nvidia_input_text})
            st.write(response["answer"])
            st.write(f"Response time: {time.process_time() - start_time} seconds")

            with st.expander("NVIDIA Document Similarity Search"):
                for i, doc in enumerate(response["context"]):
                    st.write(doc.page_content)
                    st.write("--------------------------------")
        else:
            st.error("NVIDIA Vector database is not loaded. Please generate or load the vector database first.")

elif model_option == "Ollama (local)":
    # --- Ollama Specific UI ---
    st.subheader("Ollama (local) Model Operations (gemma2:2b)")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Save Ollama Vector DB"):
            save_vector_db(OLLAMA_VECTORDIR, st.session_state.get("ollama_vectors"))
    with col2:
        if st.button("Load Ollama Vector DB"):
            st.session_state.ollama_embeddings = OllamaEmbeddings(model='gemma2:2b')
            st.session_state.ollama_vectors = load_vector_db(OLLAMA_VECTORDIR, st.session_state.ollama_embeddings)
    with col3:
        if st.button("Generate Ollama Embeddings & Questions"):
            with st.spinner("Generating Ollama Embeddings and Questions..."):
                st.session_state.ollama_embeddings = OllamaEmbeddings(model="gemma2:2b")
                text_loader = DirectoryLoader(OLLAMA_DOC_DIR, glob="**/*.txt", loader_cls=TextLoader)
                pdf_loader = PyPDFDirectoryLoader(OLLAMA_DOC_DIR)
                st.session_state.ollama_docs = text_loader.load() + pdf_loader.load()
                st.session_state.ollama_text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
                st.session_state.ollama_final_documents = st.session_state.ollama_text_splitter.split_documents(st.session_state.ollama_docs)
                st.session_state.ollama_vectors = FAISS.from_documents(st.session_state.ollama_final_documents, st.session_state.ollama_embeddings)
                st.success("Ollama Vector Store DB is ready.")

                st.session_state.ollama_generated_questions = []
                for doc in st.session_state.ollama_final_documents:
                    question = ollama_question_generation_chain.invoke({"text": doc.page_content})
                    st.session_state.ollama_generated_questions.append({"question": question.strip()})
                st.success("Ollama Questions Generated!")

    st.markdown("---")
    st.subheader("Generated Questions by Ollama:")
    if "ollama_generated_questions" in st.session_state:
        st.code(json.dumps(st.session_state.ollama_generated_questions, indent=2), language="json")
    else:
        st.info("Generate Ollama embeddings to see the questions.")

    def process_ollama_bulk_questions(questions_list):
        """Processes multiple questions for Ollama model from a list of dictionaries."""
        responses_with_references = []
        if "ollama_vectors" not in st.session_state:
            st.error("Ollama Vector database is not loaded.")
            return None
        retriever = st.session_state.ollama_vectors.as_retriever()
        qa = RetrievalQA.from_chain_type(llm=ollama_llm, chain_type="stuff", retriever=retriever)

        for item in questions_list:
            if "question" in item:
                question = item["question"]
                result = qa.run(question)
                responses_with_references.append({"question": question, "answer": result})
            else:
                st.warning(f"Skipping item due to missing 'question' key: {item}")
        return responses_with_references

    if st.button("Get Ollama Answers for Generated Questions in PDF"):
        if "ollama_vectors" in st.session_state and "ollama_generated_questions" in st.session_state:
            try:
                responses = process_ollama_bulk_questions(st.session_state.ollama_generated_questions)
                if responses:
                    create_pdf_report(responses, filename="ollama_generated_questions_answers.pdf")
                    with open("ollama_generated_questions_answers.pdf", "rb") as file:
                        st.download_button(
                            label="Download Ollama PDF Report",
                            data=file,
                            file_name="ollama_generated_questions_answers.pdf",
                            mime="application/pdf",
                        )
                else:
                    st.warning("No valid questions generated by Ollama.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.error("Ollama Vector database and/or questions are not generated. Please generate embeddings first.")

    st.markdown("---")
    ollama_input_text = st.text_input("Or ask a single question to Ollama:")

    if ollama_input_text:
        if "ollama_vectors" in st.session_state:
            retriever = st.session_state.ollama_vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, ollama_qa_chain)
            response = retrieval_chain.invoke({"input": ollama_input_text})
            st.write(response["answer"])

            with st.expander("Ollama Document Similarity Search"):
                for i, doc in enumerate(response["context"]):
                    st.write(f"**Document:** {doc.metadata.get('source', 'N/A')}, Page: {doc.metadata.get('page', 'N/A')}")
                    st.write(doc.page_content)
                    st.write("--------------------------------")
        else:
            st.error("Ollama Vector database is not loaded. Please generate embeddings first.")

# --- Phi Data Explanation ---
st.markdown("---")
st.subheader("About Phi Models and Potential Integration")
st.write(
    """
    **Phi models** are a series of small language models developed by Microsoft, focusing on reasoning and code generation capabilities.
    They are known for achieving strong performance with relatively fewer parameters, making them efficient to run.

    **Can we use Phi models here?**

    Yes, it's definitely possible to integrate Phi models into this application! Here's how it could be done:

    1. **Choose a Phi Model:** Select a specific Phi model (e.g., Phi-2, Phi-3) that suits the task.
    2. **Embedding and Language Model:**
        * **Embeddings:** You would need to use an embedding model compatible with Phi. The `SentenceTransformers` library offers several options, and some cloud providers might offer embedding APIs for Phi.
        * **Language Model:**  Accessing and using the Phi model would depend on its availability. You might use libraries like `transformers` from Hugging Face or cloud-based APIs if Microsoft Azure offers direct access.
    3. **Vector Database:** The FAISS vector database used here is model-agnostic and would work fine with embeddings generated by a Phi-compatible model.
    4. **Prompt Engineering:** You would likely need to create prompts tailored to the specific capabilities and expected input/output format of the chosen Phi model.
    5. **Integration:** You would add a new section to the Streamlit app, similar to the "NVIDIA" and "Ollama" sections, with options to:
        * Generate embeddings using a Phi-compatible embedding model.
        * Load a Phi vector database.
        * Ask questions, using the Phi language model for answering.

    **Why consider Phi?**

    * **Reasoning and Code:** If your documents contain code snippets or require more logical reasoning, Phi models might excel.
    * **Efficiency:** Smaller Phi models can be run on less powerful hardware compared to larger models.

    **Challenges of Integrating Phi:**

    * **Availability:** Accessing and using Phi models might require specific setups or cloud provider access.
    * **Ecosystem Maturity:** The Langchain integrations and community support for Phi might be less mature compared to more widely used models.

    **In summary, integrating Phi is feasible and could offer unique benefits, especially if your data involves code or complex reasoning. It would involve setting up the necessary embedding and language model components and tailoring the application accordingly.**
    """
)