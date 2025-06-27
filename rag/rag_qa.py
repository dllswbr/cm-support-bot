import numpy as np
import rag.globals as g
import re
import os
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Initialize local LLM pipeline (Mistral-7B-Instruct as default)
local_llm = None
def get_local_llm():
    global local_llm
    if local_llm is None:
        model_name = os.getenv("LOCAL_LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        local_llm = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=128)
    return local_llm

def call_local_llm(context, question):
    llm = get_local_llm()
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer concisely:"
    result = llm(prompt, do_sample=False)
    return result[0]["generated_text"].split("Answer concisely:")[-1].strip()

def call_openai(context, question):
    import openai
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    openai.api_key = api_key
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer concisely:"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=100,
        temperature=0.2
    )
    return response.choices[0].text.strip()

def extract_relevant_sentence(chunk, question):
    # Split chunk into sentences
    sentences = re.split(r'(?<=[.!?])\s+', chunk)
    if not sentences:
        return chunk
    # Use embeddings to rank sentences by similarity to question
    q_emb = g.model.encode(question)
    sent_embs = g.model.encode(sentences)
    sims = np.dot(sent_embs, q_emb) / (np.linalg.norm(sent_embs, axis=1) * np.linalg.norm(q_emb) + 1e-8)
    # Sort by similarity, filter out questions unless only option
    sorted_idx = np.argsort(sims)[::-1]
    for i in sorted_idx:
        s = sentences[i].strip()
        # If the most relevant sentence is a question, try to return the next sentence as the answer
        if s.endswith('?') or s.lower().startswith('q:'):
            if i+1 < len(sentences):
                next_s = sentences[i+1].strip()
                # If next sentence looks like an answer, return it
                if next_s and not next_s.endswith('?'):
                    return next_s
        if not s.endswith('?') and not s.lower().startswith('q:'):
            return s
    # If all are questions, return the top one
    return sentences[sorted_idx[0]].strip()

def is_unsatisfactory(answer):
    if not answer or len(answer) < 20:
        return True
    if "i don't know" in answer.lower() or "not sure" in answer.lower():
        return True
    return False

def answer_question(question, top_k=1):
    if g.index is None or g.model is None:
        return "Knowledge base not loaded. Please ingest PDFs first."
    q_emb = g.model.encode(question).astype('float32')
    D, I = g.index.search(np.expand_dims(q_emb, axis=0), top_k)
    answers = [g.chunk_id_to_text[i] for i in I[0] if i in g.chunk_id_to_text]
    if answers:
        context = answers[0]
        # Try local LLM first
        local_answer = call_local_llm(context, question)
        if not is_unsatisfactory(local_answer):
            return local_answer + f"\n\n[Source: {extract_relevant_sentence(context, question)}]"
        # Fallback to OpenAI if available
        openai_answer = call_openai(context, question)
        if openai_answer:
            return openai_answer + f"\n\n[Source: {extract_relevant_sentence(context, question)}]"
        # Fallback to extractive
        return extract_relevant_sentence(context, question)
    return "Sorry, I could not find an answer."
