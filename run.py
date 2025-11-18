"""
EduRAG - Knowledge management and hyper-personalizing educational system
Run script to start the main application
"""

import uvicorn

if __name__ == "__main__":
    print("Starting EduRAG application...")
    print("API documentation will be available at http://localhost:8000/docs")
    uvicorn.run("main_logic:app", host="0.0.0.0", port=8000, reload=True) 