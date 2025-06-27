import streamlit as st
from rag.pdf_ingest import ingest_pdfs
from rag.rag_qa import answer_question
import os

def main():
    st.title("CompanyMileage AI Support Bot")
    if 'ingested' not in st.session_state:
        with st.spinner('Ingesting PDFs...'):
            ingest_pdfs('data')
        st.session_state['ingested'] = True
        st.success('PDFs ingested!')

    st.write("Ask a question about CompanyMileage policies or support:")
    question = st.text_input("Your question:", "How do I close the pay period?")
    if st.button("Get Answer") or question:
        with st.spinner('Thinking...'):
            answer = answer_question(question)
        st.markdown(f"**Answer:** {answer}")

if __name__ == "__main__":
    main()
