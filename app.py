import streamlit as st
from PyPDF2 import PdfReader

st.title("PDF RAG App")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file is not None:
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    st.write("PDF Text Extracted:")
    st.write(text[:500])  # 앞부분만 출력
