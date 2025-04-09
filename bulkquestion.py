

import os
import json
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
from langchain.chains import RetrievalQA
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import pandas as pd
load_dotenv()

# Set up environment variables
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
# os.environ["LANGCHAIN_TRACING_V2"] = "false"
# os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# Directories for vector database and documents
VECTORDIR = r"F:\OsBook\unix\unix_vec"

# Define prompt template
prompt = ChatPromptTemplate.from_template(
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

# Initialize Gemma model
llm = Ollama(model="gemma2:2b")
output_parser = StrOutputParser()
qa_chain = prompt | llm | output_parser

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
        text_loader = DirectoryLoader(r"F:\OsBook\unix", glob="**/*.txt", loader_cls=TextLoader)
        pdf_loader = PyPDFDirectoryLoader(r"F:\OsBook\unix")
        # Add more loaders if needed (e.g., for .docx, .pptx)
        st.session_state.docs = text_loader.load() + pdf_loader.load()
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=700,chunk_overlap=50)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs)
        print("Generating Embeddings...")
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)
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
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

        for item in questions:
            if "question" in item:
                question = item["question"]
                result = qa.run(question)
                responses_with_references.append({"question": question, "answer": result})
            else:
                st.warning(f"Skipping item due to missing 'question' key: {item}")
        return responses_with_references
    except json.JSONDecodeError:
        st.error("Invalid JSON format. Please ensure your JSON is well-formed.")
        return None


def create_pdf_report(responses, filename="unix_ans.pdf"):
    """Generates a PDF report with questions, answers, and references."""
    doc = SimpleDocTemplate(filename, pagesize=letter,allow_dangerous_deserialization=True)
    styles = getSampleStyleSheet()
    flowables = []

    for response in responses:
        question_text = Paragraph(f"<b>Question:</b> {response['question']}", styles['Normal'])
        answer_text = Paragraph(f"<b>Answer:</b> {response['answer']}", styles['Normal'])
        flowables.append(question_text)
        flowables.append(answer_text)
        flowables.append(Spacer(1, 12))  # Add some space between questions

    doc.build(flowables)
    st.success(f"PDF report '{filename}' generated successfully!")

st.title("Ask & Learn with Gemma")

# Buttons for vector DB management
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
    questions=[
  {
    "question": "Why do general-purpose operating systems use monolithic kernel architecture?"
  },
  {
    "question": "Explain various services provided by the operating system."
  },
  {
    "question": "Explain the primary functions of an operating system."
  },
  {
    "question": "Differentiate multi-tasking and multi-threaded operating systems."
  },
  {
    "question": "Differentiate multi-threaded and multi-processing operating systems."
  },
  {
    "question": "What are the main differences between monolithic and microkernels?"
  },
  {
    "question": "List and explain features of the Unix operating system."
  },
  {
    "question": "Explain the Unix operating system kernel structure."
  },
  {
    "question": "Explain concepts of redirection with an example."
  },
  {
    "question": "Write and explain brelease() algorithm."
  },
  {
    "question": "What is the role of the buffer cache in the Unix operating system?"
  },
  {
    "question": "Write and explain bwrite() algorithm."
  },
  {
    "question": "Explain all scenarios for retrieval of buffer."
  },
  {
    "question": "List the advantages and limitations of buffer cache."
  },
  {
    "question": "How does Unix convert a file path into an inode?"
  },
  {
    "question": "Explain the structure of the buffer pool."
  },
  {
    "question": "List and explain the contents of the buffer header."
  },
  {
    "question": "What is the role of the buffer cache in Unix? How does it improve file system performance?"
  },
  {
    "question": "The size of each block on the hard disk is 1 KB, and the size of each inode is 128 bytes. Calculate how many inodes can fit in a single block. Additionally, if you want to create 32 files, determine the number of blocks required to store the inodes for all these files."
  },
  {
    "question": "In a file system, each inode occupies 128 bytes, and the block size is 2 KB. Calculate the number of inodes that can be stored in a single block. If a directory contains 100 files, determine how many blocks are required to store the inodes for these files."
  },
  {
    "question": "The size of each block on the hard disk is 1 KB, and the size of each inode is 128 bytes. Start address of Inode block 3. In which block inode 15 will be found?"
  },
  {
    "question": "A regular file is 25 KB. Analyze how the 13-member array handles this allocation using direct, single-indirect, and double-indirect pointers. Provide a detailed breakdown of the allocation process. Consider block size is 1 KB."
  },
  {
    "question": "If a file is 15 KB in size and the block size is 1 KB, determine how many blocks are allocated to direct pointers and how many are allocated through single-indirect pointers. Explain your reasoning."
  },
  {
    "question": "Explain the role of single-indirect pointers when storing a file of 20 KB using a 13-member array in the inode with a block size of 1 KB. How many blocks are utilized?"
  },
  {
    "question": "A file system uses a free block list to allocate disk blocks. Describe how the alloc algorithm assigns blocks to a new file requiring 5 blocks. If the free block list initially contains [5, 12, 20, 25, 30, 32, 47], show the final state of the free block list after allocation."
  },
  {
    "question": "A file is deleted, and the free algorithm is used to reclaim its blocks. The deleted file occupied disk blocks [10, 11, 15, 20]. If the free block list initially contains [5, 6, 8], demonstrate the state of the list after the blocks are reclaimed."
  },
  {
    "question": "A file system uses an indexed free block structure. If the free block list has [100, 105, 110, 120, 130], explain how the alloc algorithm assigns blocks to a new file of size 3 KB (block size 1 KB). Show the updated free block list."
  },
  {
    "question": "Explain the three standard input/output (I/O) streams in Unix systems. Briefly describe the purpose of each stream."
  },
  {
    "question": "Differentiate between hardware interrupts and software exceptions with examples of how each is triggered in a system."
  },
  {
    "question": "Discuss the concept of interruptions and their types: maskable and non-maskable interrupts, high-priority and low-priority interrupts. Provide an example of each type."
  },
  {
    "question": "List and explain the contents of the process table."
  },
  {
    "question": "List and explain the contents of Uarea."
  },
  {
    "question": "Draw the process state transition diagram and explain each state of the process."
  },
  {
    "question": "List and explain the contents of the context of a process."
  },
  {
    "question": "Write and explain an algorithm to handle interrupts."
  },
  {
    "question": "Apply the shortest job first algorithm on the given data, draw the Gantt chart, and calculate the average waiting time, average completion time, and average turnaround time."
  },
  {
    "question": "State the reasons why the kernel swaps out the process."
  },
  {
    "question": "Explain the concept of fork swap and expansion swap."
  },
  {
    "question": "List the causes of a page fault."
  },
  {
    "question": "Demonstrate the changes made in the swap map table for a given scenario."
  },
  {
    "question": "Write and explain the algorithm for allocating space from swap map."
  },
  {
    "question": "Write and explain the algorithm for swapping out the process from the main memory to the swap area."
  }
]

except KeyError:
    st.error("questions.csv must have a 'question' column.")
    questions = []
    
    
# Input for bulk questions via JSON
bulk_input_text = st.text_area(
    "Enter multiple questions in JSON format:",
    value=json.dumps(questions, indent=2),
    height=200,
)

if st.button("Get Answers in PDF"):
    if "vectors" in st.session_state:
        try:
            responses = process_bulk_questions(bulk_input_text)
            if responses:
                create_pdf_report(responses)
                with open("answer_report.pdf", "rb") as file:
                    st.download_button(
                        label="Download PDF Report",
                        data=file,
                        file_name="answer_report.pdf",
                        mime="application/pdf",
                    )
            else:
                st.warning("No valid questions found in JSON input.")
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON format: {e}. Please ensure your JSON is well-formed.")
    else:
        st.error("Vector database is not loaded. Please generate or load the vector database first.")
        
# Single question input (optional)
st.markdown("---")
input_text = st.text_input("Or ask a single question:")

if input_text:
    if "vectors" in st.session_state:
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, qa_chain)
        response = retrieval_chain.invoke({"input": input_text})
        st.write(response["answer"])

        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.write(f"**Document:** {doc.metadata.get('source', 'N/A')}, Page: {doc.metadata.get('page', 'N/A')}")
                st.write(doc.page_content)
                st.write("--------------------------------")
    else:
        st.error("Vector database is not loaded. Please generate or load the vector database first.")

