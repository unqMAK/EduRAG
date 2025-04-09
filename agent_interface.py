
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
NVIDIA_VECTORDIR = r"D:\MCA\KrishNaik's_course\Nvidia-NIM-main\Nvidia-NIM-main\us_census\Nvidia_vector"
OLLAMA_VECTORDIR = r"D:\MCA\KrishNaik's_course\Nvidia-NIM-main\Nvidia-NIM-main\us_census\ollama_vector"
DOCUMENT_DIR = r"D:\MCA\KrishNaik's_course\Nvidia-NIM-main\Nvidia-NIM-main\us_census"
OLLAMA_DOC_DIR = r"D:\MCA\KrishNaik's_course\Nvidia-NIM-main\Nvidia-NIM-main\us_census" # Adjust this path

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

def process_nvidia_bulk_questions(questions, nvidia_vectors):
    """Processes multiple questions for NVIDIA model."""
    responses_with_references = []
    if not nvidia_vectors:
        st.error("NVIDIA Vector database is not loaded.")
        return None
    retriever = nvidia_vectors.as_retriever()
    llm = ChatNVIDIA(model="meta/llama3-70b-instruct")
    document_chain = create_stuff_documents_chain(llm, nvidia_prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    for question_data in questions:
        question = question_data.get("question")
        if question:
            response = retrieval_chain.invoke({"input": question})
            responses_with_references.append({"question": question, "answer": response["answer"]})
        else:
            st.warning(f"Skipping item due to missing 'question' key: {question_data}")
    return responses_with_references

def process_ollama_bulk_questions(questions_list, ollama_vectors):
    """Processes multiple questions for Ollama model."""
    responses_with_references = []
    if not ollama_vectors:
        st.error("Ollama Vector database is not loaded.")
        return None
    retriever = ollama_vectors.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=ollama_llm, chain_type="stuff", retriever=retriever)

    for item in questions_list:
        if "question" in item:
            question = item["question"]
            result = qa.run(question)
            responses_with_references.append({"question": question, "answer": result})
        else:
            st.warning(f"Skipping item due to missing 'question' key: {item}")
    return responses_with_references

# --- Streamlit UI ---
st.title("Agentic AI Interface")

tab_nvidia, tab_ollama, tab_control = st.tabs(["NVIDIA Agent", "Ollama Agent", "Control Panel"])

# --- NVIDIA Agent Tab ---
with tab_nvidia:
    st.subheader("NVIDIA Agent Controls")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Save NVIDIA Vector DB", key="nvidia_save"):
            save_vector_db(NVIDIA_VECTORDIR, st.session_state.get("nvidia_vectors"))
    with col2:
        if st.button("Load NVIDIA Vector DB", key="nvidia_load"):
            st.session_state.nvidia_embeddings = NVIDIAEmbeddings()
            st.session_state.nvidia_vectors = load_vector_db(NVIDIA_VECTORDIR, st.session_state.nvidia_embeddings)
    with col3:
        if st.button("Generate NVIDIA Embeddings", key="nvidia_gen"):
            with st.spinner("Generating NVIDIA Embeddings..."):
                st.session_state.nvidia_embeddings = NVIDIAEmbeddings()
                text_loader = DirectoryLoader(DOCUMENT_DIR, glob="**/*.txt", loader_cls=TextLoader)
                pdf_loader = PyPDFDirectoryLoader(DOCUMENT_DIR)
                docs = text_loader.load() + pdf_loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
                final_documents = text_splitter.split_documents(docs)
                st.session_state.nvidia_vectors = FAISS.from_documents(final_documents, st.session_state.nvidia_embeddings)
                st.success("NVIDIA Vector Store DB is ready.")

    st.markdown("---")
    nvidia_input_text = st.text_area("Ask NVIDIA a question:", key="nvidia_single_input")
    if st.button("Ask NVIDIA", key="nvidia_ask_single"):
        if "nvidia_vectors" in st.session_state:
            llm = ChatNVIDIA(model="meta/llama3-70b-instruct")
            document_chain = create_stuff_documents_chain(llm, nvidia_prompt)
            retriever = st.session_state.nvidia_vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            start_time = time.process_time()
            response = retrieval_chain.invoke({"input": nvidia_input_text})
            st.write(f"**NVIDIA's Answer:** {response['answer']}")
            st.write(f"Response time: {time.process_time() - start_time} seconds")

            with st.expander("NVIDIA Document Similarity Search"):
                for i, doc in enumerate(response["context"]):
                    st.write(doc.page_content)
                    st.write("--------------------------------")
        else:
            st.error("NVIDIA Vector database is not loaded. Please generate or load the vector database first.")

# --- Ollama Agent Tab ---
with tab_ollama:
    st.subheader("Ollama Agent Controls (gemma2:2b)")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("Save Ollama Vector DB", key="ollama_save"):
            save_vector_db(OLLAMA_VECTORDIR, st.session_state.get("ollama_vectors"))
    with col2:
        if st.button("Load Ollama Vector DB", key="ollama_load"):
            st.session_state.ollama_embeddings = OllamaEmbeddings(model='gemma2:2b')
            st.session_state.ollama_vectors = load_vector_db(OLLAMA_VECTORDIR, st.session_state.ollama_embeddings)
            if "ollama_vectors" in st.session_state and st.session_state.ollama_vectors is not None:
                st.session_state.ollama_embeddings_generated = True
                if "ollama_final_documents" not in st.session_state:
                    text_loader = DirectoryLoader(OLLAMA_DOC_DIR, glob="**/*.txt", loader_cls=TextLoader)
                    pdf_loader = PyPDFDirectoryLoader(OLLAMA_DOC_DIR)
                    st.session_state.ollama_docs = text_loader.load() + pdf_loader.load()
                    st.session_state.ollama_text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
                    st.session_state.ollama_final_documents = st.session_state.ollama_text_splitter.split_documents(st.session_state.ollama_docs)

    with col3:
        if st.button("Generate Ollama Embeddings", key="ollama_gen_embeddings"):
            with st.spinner("Generating Ollama Embeddings..."):
                st.session_state.ollama_embeddings = OllamaEmbeddings(model="gemma2:2b")
                text_loader = DirectoryLoader(OLLAMA_DOC_DIR, glob="**/*.txt", loader_cls=TextLoader)
                pdf_loader = PyPDFDirectoryLoader(OLLAMA_DOC_DIR)
                st.session_state.ollama_docs = text_loader.load() + pdf_loader.load()
                st.session_state.ollama_text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
                st.session_state.ollama_final_documents = st.session_state.ollama_text_splitter.split_documents(st.session_state.ollama_docs)
                st.session_state.ollama_embeddings_generated = True
                st.success("Ollama Vector Store DB is ready.")

    with col4:
        if st.session_state.get("ollama_embeddings_generated", False):
            if st.button("Generate Ollama Questions", key="ollama_gen_questions"):
                if "ollama_final_documents" in st.session_state:
                    with st.spinner("Generating Ollama Questions..."):
                        st.session_state.ollama_generated_questions = []
                        ollama_question_generation_chain = ollama_question_generation_prompt | ollama_llm
                        progress_bar = st.progress(0)
                        num_documents = len(st.session_state.ollama_final_documents)

                        for i, doc in enumerate(st.session_state.ollama_final_documents):
                            with st.spinner(f"Generating question from document {i+1}/{num_documents}..."): # Spinner per document
                                question = ollama_question_generation_chain.invoke({"text": doc.page_content})
                                question_text = question.strip()
                                source_file = os.path.basename(doc.metadata.get('source', 'Document')) # Get filename
                                st.session_state.ollama_generated_questions.append({"question": question_text, "source": source_file}) # Store source

                                progress_percent = (i + 1) / num_documents
                                progress_bar.progress(progress_percent)

                        progress_bar.empty()
                        st.success("Ollama Questions Generated!")
                        # Update the text area with generated questions only AFTER generation is complete
                        st.session_state.ollama_generated_questions_text = json.dumps(st.session_state.ollama_generated_questions, indent=2)


                else:
                    st.error("Error: Document chunks not found. Please generate Ollama Embeddings first.")
        elif not st.session_state.get("ollama_embeddings_generated", False) and "ollama_vectors" not in st.session_state:
            st.info("Generate Ollama Embeddings first to generate questions.")

    st.markdown("---")
    if "ollama_generated_questions" in st.session_state:
        st.subheader("Generated Questions by Ollama:")
        st.session_state.ollama_generated_questions_text = st.text_area(
            "Ollama Generated Questions (JSON):",
            value=st.session_state.get("ollama_generated_questions_text", ""), # Use session state value, default to empty string
            height=200,
            key="ollama_generated_text"
        )
        if st.button("Save Questions as JSON", key="save_questions_json"):
            json_data = json.dumps(st.session_state.ollama_generated_questions, indent=4)
            st.download_button(
                label="Download Questions JSON",
                data=json_data,
                file_name="ollama_generated_questions.json",
                mime="application/json"
            )
    else:
        st.info("Generate Ollama embeddings and questions to see the questions.")

    st.markdown("---")
    ollama_input_text = st.text_input("Ask Ollama a question:", key="ollama_single_input")
    if st.button("Ask Ollama", key="ollama_ask_single"):
        if "ollama_vectors" in st.session_state:
            retriever = st.session_state.ollama_vectors.as_retriever()
            ollama_qa_chain = ollama_qa_prompt | ollama_llm
            retrieval_chain = create_retrieval_chain(retriever, ollama_qa_chain)
            response = retrieval_chain.invoke({"input": ollama_input_text})
            st.write(f"**Ollama's Answer:** {response['answer']}")

            with st.expander("Ollama Document Similarity Search"):
                for i, doc in enumerate(response["context"]):
                    st.write(f"**Document:** {doc.metadata.get('source', 'N/A')}, Page: {doc.metadata.get('page', 'N/A')}")
                    st.write(doc.page_content)
                    st.write("--------------------------------")
        else:
            st.error("Ollama Vector database is not loaded. Please generate embeddings first.")

# --- Control Panel Tab ---
with tab_control:
    st.subheader("Agent Coordination & Control")

    # Trigger Ollama to generate questions (using existing embeddings if available)
    if st.button("Trigger Ollama to Generate Questions"):
        if "ollama_vectors" in st.session_state: # Check if vectors exist, assume embeddings are generated if vectors exist
            with st.spinner("Ollama is generating questions..."):
                st.session_state.ollama_generated_questions = []
                ollama_question_generation_chain = ollama_question_generation_prompt | ollama_llm
                progress_bar = st.progress(0)
                for i, doc in enumerate(st.session_state.ollama_final_documents):
                    question = ollama_question_generation_chain.invoke({"text": doc.page_content})
                    st.session_state.ollama_generated_questions.append({"question": question.strip()})
                    progress_bar.progress((i+1) / len(st.session_state.ollama_final_documents))
                progress_bar.empty()
                st.success("Ollama has generated questions.")
        else:
            st.error("Ollama embeddings are not generated yet. Please generate Ollama embeddings first in Ollama Agent tab.")


    # Option to use Ollama's generated questions for NVIDIA
    if "ollama_generated_questions" in st.session_state:
        st.markdown("---")
        st.write("Use Ollama's generated questions for NVIDIA:")
        use_ollama_questions = st.checkbox("Use Ollama Generated Questions", key="use_ollama_questions")
        if use_ollama_questions:
            if "ollama_generated_questions" in st.session_state:
                questions_for_nvidia = st.session_state.ollama_generated_questions
            else:
                st.warning("No questions generated by Ollama yet.")
                questions_for_nvidia = []

            if st.button("Trigger NVIDIA to Answer Ollama's Questions"):
                if "nvidia_vectors" in st.session_state:
                    with st.spinner("NVIDIA is answering Ollama's questions..."):
                        responses = process_nvidia_bulk_questions(questions_for_nvidia, st.session_state.nvidia_vectors)
                        if responses:
                            st.subheader("NVIDIA's Answers to Ollama's Questions:")
                            for resp in responses:
                                st.write(f"**Question:** {resp['question']}")
                                st.write(f"**Answer:** {resp['answer']}")
                                st.markdown("---")
                        else:
                            st.warning("NVIDIA could not answer the questions.")
                else:
                    st.error("NVIDIA Vector database is not loaded.")

    st.markdown("---")
    st.subheader("Bulk Question Answering (Manual Input)")
    bulk_input_mode = st.radio("Select agent for bulk question answering:", ["NVIDIA", "Ollama"], key="bulk_mode")
    bulk_input_text = st.text_area(f"Enter multiple questions in JSON format for {bulk_input_mode}:", key="bulk_input", height=150)

    if st.button(f"Get Bulk Answers with {bulk_input_mode}"):
        try:
            bulk_questions = json.loads(bulk_input_text)
            if not isinstance(bulk_questions, list):
                st.error("Invalid JSON format. Please provide a list of questions.")
            else:
                with st.spinner(f"{bulk_input_mode} is processing bulk questions..."):
                    if bulk_input_mode == "NVIDIA":
                        if "nvidia_vectors" in st.session_state:
                            responses = process_nvidia_bulk_questions(bulk_questions, st.session_state.nvidia_vectors)
                            if responses:
                                create_pdf_report(responses, "nvidia_bulk_answers.pdf")
                                with open("nvidia_bulk_answers.pdf", "rb") as file:
                                    st.download_button(
                                        label="Download NVIDIA Bulk Answers PDF",
                                        data=file,
                                        file_name="nvidia_bulk_answers.pdf",
                                        mime="application/pdf",
                                    )
                        else:
                            st.error("NVIDIA Vector database is not loaded.")
                    elif bulk_input_mode == "Ollama":
                        if "ollama_vectors" in st.session_state:
                            responses = process_ollama_bulk_questions(bulk_questions, st.session_state.ollama_vectors)
                            if responses:
                                create_pdf_report(responses, "ollama_bulk_answers.pdf")
                                with open("ollama_bulk_answers.pdf", "rb") as file:
                                    st.download_button(
                                        label="Download Ollama Bulk Answers PDF",
                                        data=file,
                                        file_name="ollama_bulk_answers.pdf",
                                        mime="application/pdf",
                                    )
                        else:
                            st.error("Ollama Vector database is not loaded.")
        except json.JSONDecodeError:
            st.error("Invalid JSON format. Please ensure your JSON is well-formed.")