from pypdf import PdfReader
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import rag.globals as g
import re

# Helper to split text into chunks
def split_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i+chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def extract_qa_pairs(text):
    # Split text into Q&A pairs using regex
    qa_pattern = re.compile(r'(Q:.*?)(?=Q:|\Z)', re.DOTALL | re.IGNORECASE)
    pairs = qa_pattern.findall(text)
    return [p.strip() for p in pairs if p.strip()]

def ingest_pdfs(pdf_dir):
    print(f"Ingesting PDFs from {pdf_dir}...")
    g.model = SentenceTransformer('all-MiniLM-L6-v2')
    all_embeddings = []
    g.chunk_id_to_text.clear()
    chunk_id = 0
    for filename in os.listdir(pdf_dir):
        if filename.endswith('.pdf'):
            path = os.path.join(pdf_dir, filename)
            reader = PdfReader(path)
            text = "\n".join(page.extract_text() or '' for page in reader.pages)
            qa_chunks = extract_qa_pairs(text)
            if qa_chunks:
                for chunk in qa_chunks:
                    g.chunk_id_to_text[chunk_id] = chunk
                    emb = g.model.encode(chunk)
                    all_embeddings.append(emb)
                    chunk_id += 1
                print(f"Parsed {filename} ({len(text)} characters, {len(qa_chunks)} Q&A pairs)")
            else:
                # fallback to old chunking if no Q&A found
                words = text.split()
                for i in range(0, len(words), 500):
                    chunk = ' '.join(words[i:i+500])
                    g.chunk_id_to_text[chunk_id] = chunk
                    emb = g.model.encode(chunk)
                    all_embeddings.append(emb)
                    chunk_id += 1
                print(f"Parsed {filename} ({len(text)} characters, fallback chunking)")
    if all_embeddings:
        embs = np.vstack(all_embeddings).astype('float32')
        g.index = faiss.IndexFlatL2(embs.shape[1])
        g.index.add(embs)
    print("PDF ingestion complete.")
