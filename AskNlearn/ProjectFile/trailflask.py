import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain

load_dotenv()

app = Flask(__name__)

# Global variables
DOC_DIR_TXT = None
DOC_DIR_PDF = None
VECTORDIR = None
VECTOR_DB = None

# Initialize prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    </context>
    Question: {input}
    """
)

# Initialize LLM and output parser
llm = Ollama(model="gemma2:2b")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser


# Function to generate FAISS vector database
def generate_vector_db():
    global VECTORDIR, DOC_DIR_TXT, DOC_DIR_PDF
    embeddings = OllamaEmbeddings(model="gemma2:2b")

    # Load documents dynamically
    docs = []
    if DOC_DIR_TXT:
        text_loader = DirectoryLoader(DOC_DIR_TXT, glob="**/*.txt", loader_cls=TextLoader)
        docs.extend(text_loader.load())
    if DOC_DIR_PDF:
        pdf_loader = PyPDFDirectoryLoader(DOC_DIR_PDF)
        docs.extend(pdf_loader.load())

    if not docs:
        return "No documents found to process. Please check the directory paths."

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
    final_documents = text_splitter.split_documents(docs)

    # Generate and save FAISS vectors
    vectors = FAISS.from_documents(final_documents, embeddings)
    vectors.save_local(VECTORDIR)
    return f"Vector database saved to {VECTORDIR}"


# Function to load FAISS vector database
def load_vector_db():
    global VECTOR_DB, VECTORDIR
    try:
        embeddings = OllamaEmbeddings(model="gemma2:2b")
        VECTOR_DB = FAISS.load_local(VECTORDIR, embeddings, allow_dangerous_deserialization=True)
        return "Vector database loaded successfully."
    except Exception as e:
        return f"Error loading vector database: {e}"


# Flask Routes
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/set_paths", methods=["POST"])
def set_paths():
    global DOC_DIR_TXT, DOC_DIR_PDF, VECTORDIR
    data = request.json
    DOC_DIR_TXT = data.get("text_dir", "")
    DOC_DIR_PDF = data.get("pdf_dir", "")
    VECTORDIR = data.get("vector_dir", "")

    # Validate paths
    response = {"status": "success", "warnings": []}
    if DOC_DIR_TXT and not os.path.exists(DOC_DIR_TXT):
        response["warnings"].append(f"Text directory '{DOC_DIR_TXT}' does not exist.")
    if DOC_DIR_PDF and not os.path.exists(DOC_DIR_PDF):
        response["warnings"].append(f"PDF directory '{DOC_DIR_PDF}' does not exist.")
    if not os.path.exists(VECTORDIR):
        os.makedirs(VECTORDIR, exist_ok=True)
        response["warnings"].append(f"Vector database directory '{VECTORDIR}' created.")

    return jsonify(response)


@app.route("/generate_vectors", methods=["POST"])
def generate_vectors():
    result = generate_vector_db()
    return jsonify({"message": result})


@app.route("/load_vectors", methods=["POST"])
def load_vectors():
    result = load_vector_db()
    return jsonify({"message": result})


@app.route("/ask_question", methods=["POST"])
def ask_question():
    global VECTOR_DB
    if VECTOR_DB is None:
        return jsonify({"error": "Vector database not loaded. Load it first."}), 400

    # Get the user question
    data = request.json
    user_question = data.get("question", "")

    # Set up retriever and chain
    retriever = VECTOR_DB.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, chain)

    # Process the user's question
    response = retrieval_chain.invoke({"input": user_question})
    return jsonify({
        "answer": response["answer"],
        "contexts": [doc.page_content for doc in response["context"]]
    })


if __name__ == "__main__":
    app.run(debug=True)