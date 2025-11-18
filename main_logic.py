import os
import shutil
from typing import Optional, Tuple, List

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from pymongo import MongoClient
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from pydub import AudioSegment
import moviepy.editor as mp
import speech_recognition as sr
from PIL import Image
import pytesseract

# FastAPI app setup
app = FastAPI(
    title="EduRAG",
    description="Knowledge management and hyper-personalizing educational system",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Base directory for subjects
BASE_DIR = "Subjects"
if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)

# MongoDB setup
MONGO_URI = "mongodb://localhost:27017/"
client = MongoClient(MONGO_URI)
db = client["EduRAG"]
files_collection = db["files"]

# Initialize global vector database instance
VECTOR_DB = None

# Initialize LLM
llm = Ollama(model="gemma2:2b")
output_parser = StrOutputParser()
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
chain = prompt | llm | output_parser

# --- Utility Functions ---
def convert_audio_to_wav(audio_path: str, output_path: str = "converted_audio.wav") -> Optional[str]:
    """Convert audio file to WAV format suitable for speech recognition."""
    try:
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_frame_rate(16000).set_sample_width(2).set_channels(1)
        audio.export(output_path, format="wav")
        return output_path
    except Exception as e:
        print(f"Error converting audio: {e}")
        return None

def extract_audio_from_video(video_path: str) -> Optional[str]:
    """Extract audio from video file and save it as WAV."""
    try:
        video_clip = mp.VideoFileClip(video_path)
        audio_output_path = "extracted_audio.wav"
        video_clip.audio.write_audiofile(audio_output_path)
        return audio_output_path
    except Exception as e:
        print(f"Error extracting audio from video: {e}")
        return None

def transcribe_audio(audio_path: str) -> Optional[str]:
    """Transcribe audio to text using speech recognition."""
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
        return recognizer.recognize_google(audio_data)
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return None

def extract_text_from_image(image_path: str) -> Optional[str]:
    """Extract text from image using OCR."""
    try:
        return pytesseract.image_to_string(Image.open(image_path))
    except Exception as e:
        print(f"Error extracting text from image: {e}")
        return None

def process_file(file_path: str, subject_path: str) -> str:
    """Process uploaded file based on type and extract content."""
    file_extension = os.path.splitext(file_path)[1].lower()
    text_output = None

    if file_extension in ['.mp4', '.avi', '.mov', '.mkv', '.flv']:
        audio_path = extract_audio_from_video(file_path)
        if audio_path:
            text_output = transcribe_audio(audio_path)
            # Clean up temporary audio file
            os.remove(audio_path)

    elif file_extension in ['.mp3', '.wav', '.flac']:
        if not file_path.lower().endswith('.wav'):
            converted_path = convert_audio_to_wav(file_path)
            if converted_path:
                text_output = transcribe_audio(converted_path)
                os.remove(converted_path)
        else:
            text_output = transcribe_audio(file_path)

    elif file_extension in ['.png', '.jpg', '.jpeg']:
        text_output = extract_text_from_image(file_path)

    elif file_extension == '.pdf':
        pdf_folder = os.path.join(subject_path, "pdf")
        os.makedirs(pdf_folder, exist_ok=True)
        pdf_file_path = os.path.join(pdf_folder, os.path.basename(file_path))
        if not os.path.exists(pdf_file_path):  # Avoid overwriting
            shutil.move(file_path, pdf_file_path)
        return f"PDF file saved to: {pdf_file_path}"
    
    elif file_extension in ['.txt', '.md', '.doc', '.docx']:
        # If it's already text, just move it to the text folder
        text_folder = os.path.join(subject_path, "Text")
        os.makedirs(text_folder, exist_ok=True)
        text_file_path = os.path.join(text_folder, os.path.basename(file_path))
        if not os.path.exists(text_file_path):
            shutil.move(file_path, text_file_path)
        return f"Text file saved to: {text_file_path}"

    if text_output:
        text_folder = os.path.join(subject_path, "Text")
        os.makedirs(text_folder, exist_ok=True)
        text_file_path = os.path.join(text_folder, f"{os.path.splitext(os.path.basename(file_path))[0]}.txt")
        with open(text_file_path, 'w', encoding='utf-8') as f:
            f.write(text_output)
        return f"Text extracted and saved to: {text_file_path}"
    
    return "Unsupported file type or processing failed."

def generate_vector_db(subject_path: str, subject_name: str) -> Tuple[str, Optional[FAISS]]:
    """Generate vector database from text and PDF files in subject directory."""
    global VECTOR_DB

    embeddings = OllamaEmbeddings(model="gemma2:2b")
    docs = []

    # Load text and pdf files
    DOC_DIR_TXT = os.path.join(subject_path, "Text")
    DOC_DIR_PDF = os.path.join(subject_path, "pdf")

    if os.path.exists(DOC_DIR_TXT):
        text_loader = DirectoryLoader(DOC_DIR_TXT, glob="**/*.txt", loader_cls=TextLoader)
        docs.extend(text_loader.load())
    
    if os.path.exists(DOC_DIR_PDF):
        pdf_loader = PyPDFDirectoryLoader(DOC_DIR_PDF)
        docs.extend(pdf_loader.load())

    if not docs:
        return "No documents to process.", None

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=300)
    final_docs = text_splitter.split_documents(docs)

    # Generate and save FAISS vectors
    vector_folder = os.path.join(subject_path, f"{subject_name}VectorDatabase")
    os.makedirs(vector_folder, exist_ok=True)

    vectors = FAISS.from_documents(final_docs, embeddings, allow_dangerous_deserialization=True)
    vectors.save_local(vector_folder)

    VECTOR_DB = vectors
    return f"Vector database for '{subject_name}' generated and saved.", vectors

def save_to_mongo(subject: str, file_type: str, file_path: str, content: str = None, vector_db_path: str = None):
    """Save file metadata to MongoDB."""
    document = {
        "subject": subject,
        "file_type": file_type,
        "file_path": file_path,
        "content": content,
        "vector_db_path": vector_db_path
    }
    
    result = files_collection.insert_one(document)
    return result.inserted_id

def load_vector_db(subject_path: str, subject_name: str) -> Optional[FAISS]:
    """Load the vector database for a given subject."""
    global VECTOR_DB
    
    vector_folder = os.path.join(subject_path, f"{subject_name}VectorDatabase")
    
    if not os.path.exists(vector_folder):
        return None
    
    embeddings = OllamaEmbeddings(model="gemma2:2b")
    VECTOR_DB = FAISS.load_local(vector_folder, embeddings)
    return VECTOR_DB

# --- API Endpoints ---
@app.get("/")
async def home():
    """Home page that lists all subjects."""
    subjects = [f for f in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, f))]
    return {"subjects": subjects}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), subject: str = Form(...)):
    """Upload a file for a specific subject."""
    if not subject:
        raise HTTPException(status_code=400, detail="Missing subject")
    
    # Create subject directory if it doesn't exist
    subject_path = os.path.join(BASE_DIR, subject)
    os.makedirs(subject_path, exist_ok=True)
    
    # Save the uploaded file temporarily
    upload_dir = os.path.join("uploads", subject)
    os.makedirs(upload_dir, exist_ok=True)
    
    temp_file_path = os.path.join(upload_dir, file.filename)
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process the file based on its type
    file_type = os.path.splitext(file.filename)[1].lower()
    process_result = process_file(temp_file_path, subject_path)
    
    # Generate or update vector database
    vector_db_result, _ = generate_vector_db(subject_path, subject)
    
    # Save file metadata to MongoDB
    file_id = save_to_mongo(
        subject=subject,
        file_type=file_type,
        file_path=os.path.join(subject_path, file.filename),
        vector_db_path=os.path.join(subject_path, f"{subject}VectorDatabase")
    )
    
    return {
        "message": process_result,
        "vector_db_message": vector_db_result,
        "file_id": str(file_id)
    }

@app.post("/ask")
async def ask_question(request: dict):
    """Ask a question about a specific subject."""
    subject = request.get("subject")
    question = request.get("question")
    
    if not subject or not question:
        raise HTTPException(status_code=400, detail="Subject and question are required")
    
    subject_path = os.path.join(BASE_DIR, subject)
    if not os.path.exists(subject_path):
        raise HTTPException(status_code=404, detail=f"Subject '{subject}' not found")
    
    # Load or use existing vector database
    global VECTOR_DB
    if VECTOR_DB is None:
        VECTOR_DB = load_vector_db(subject_path, subject)
    
    if VECTOR_DB is None:
        raise HTTPException(status_code=404, detail=f"Vector database for subject '{subject}' not found")
    
    # Query the vector database
    retriever = VECTOR_DB.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, chain)
    response = retrieval_chain.invoke({"input": question})
    
    return {
        "answer": response["answer"],
        "contexts": [doc.page_content for doc in response["context"]]
    }

@app.delete("/files/{file_id}")
async def delete_file(file_id: str):
    """Delete a file from the system."""
    file = files_collection.find_one({"_id": file_id})
    if not file:
        raise HTTPException(status_code=404, detail="File not found")
    
    # Delete file from filesystem if it exists
    if os.path.exists(file["file_path"]):
        os.remove(file["file_path"])
    
    # Delete file metadata from database
    result = files_collection.delete_one({"_id": file_id})
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=500, detail="Failed to delete file")
    
    # Regenerate vector database for the subject
    subject_path = os.path.join(BASE_DIR, file["subject"])
    vector_db_result, _ = generate_vector_db(subject_path, file["subject"])
    
    return {
        "message": "File deleted successfully",
        "vector_db_message": vector_db_result
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_logic:app", host="0.0.0.0", port=8000, reload=True) 