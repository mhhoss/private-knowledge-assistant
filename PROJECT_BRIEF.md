# Project: Private Knowledge Assistant (RAG)

## Goal
Build a system where users can upload PDF/DOCX files and ask questions. 
The AI must answer ONLY based on the uploaded documents and cite the source.

## Core Features
- Upload multiple PDF and DOCX files
- Process and chunk documents
- Store embeddings in Chroma
- Chat interface to ask questions
- Show source (filename + page/chunk) with every answer
- Simple and clean UI

## Tech Stack
- Python 3.11+
- LlamaIndex
- Chroma
- FastAPI (backend)
- Streamlit (frontend)
- OpenAI gateway (gpt-4o-mini + text-embedding-3-small)
- pypdf, python-docx

## Requirements
- Must work offline after documents are processed (except LLM calls)
- Clean project structure
- Easy to run with one command
- Good error handling
- README with clear instructions