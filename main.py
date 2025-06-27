import os
from rag.pdf_ingest import ingest_pdfs
from rag.rag_qa import answer_question

def main():
    print("Welcome to the CompanyMileage AI Support Bot!")
    ingest_pdfs('data')
    while True:
        question = input("Ask a question (or type 'exit'): ")
        if question.lower() == 'exit':
            break
        answer = answer_question(question)
        print(f"Answer: {answer}\n")

if __name__ == "__main__":
    main()
