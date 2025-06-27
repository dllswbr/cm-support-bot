# CompanyMileage AI Support Bot

This project is an AI support bot for CompanyMileage using Retrieval-Augmented Generation (RAG) with PDF document ingestion. It is designed to answer questions based on company documents provided in PDF format.

## Features
- PDF ingestion and parsing
- Embedding and vector storage for document retrieval
- Simple chatbot interface (CLI or web)
- Easy to deploy and extend

## Getting Started
1. Ensure you have Python 3.9+ installed.
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Place your PDF documents in the `data/` folder.
4. Run the bot:
   ```sh
   python main.py
   ```

## Project Structure
- `main.py` - Entry point for the chatbot
- `rag/` - RAG logic and utilities
- `data/` - PDF documents
- `.github/copilot-instructions.md` - Copilot custom instructions

## License
MIT License
