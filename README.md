# EduRAG - FastAPI Version

EduRAG is a knowledge management and hyper-personalizing educational system, where students can ask questions about course materials, request explanations of topics, ask for model question papers, or upload question banks and get model answers.

## Features

- **AI-Powered Learning:** Advanced AI technology provides personalized assistance and instant answers to student questions
- **Multi-format Support:** Upload and process various file formats (PDF, text, audio, video, images)
- **Subject Management:** Organize content by subjects for easy access and retrieval
- **User Roles:** Separate interfaces for students and professors with appropriate permissions
- **Vector Database:** Utilizes FAISS vector database for efficient semantic search and retrieval

## Tech Stack

- **Backend:** FastAPI
- **Database:** MongoDB
- **LLM Integration:** Ollama with Gemma2:2b model
- **Vector Store:** FAISS
- **Frontend:** HTML/CSS/JavaScript with Bootstrap

## Getting Started

### Prerequisites

- Python 3.9+
- MongoDB
- Ollama with Gemma2:2b model

### Installation

1. Clone the repository
```bash
git clone https://github.com/unqMAK/EduRAG.git
cd EduRAG
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Set up environment variables
```bash
# Create .env file
touch .env

# Add the following variables
MONGO_URI=mongodb://localhost:27017/
DATABASE_NAME=EduRAG
LLM_MODEL=gemma2:2b
JWT_SECRET_KEY=your_secret_key_here
```

4. Run the application
```bash
uvicorn app.main:app --reload
```

The application will be available at [http://localhost:8000](http://localhost:8000)

## API Documentation

Once the application is running, you can access the API documentation at [http://localhost:8000/docs](http://localhost:8000/docs)

## Usage

1. Register as a student or professor
2. Login to access the system
3. Professors can create subjects and upload course materials
4. Students can select subjects and ask questions about the materials

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request 