import streamlit as st
from set_openai_api_key import set_openai_api_key
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA

# OpenAI API 키 설정
set_openai_api_key()

# Streamlit 앱 시작
st.title("PDF 기반 RAG 서비스")
st.write("PDF 파일을 업로드하고 질문을 입력하세요!")

# 파일 업로드
uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type="pdf")

if uploaded_file:
    # PDF 파일 읽기
    loader = PyPDFLoader(uploaded_file)
    documents = loader.load()

    # 벡터 스토어 생성
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(documents, embeddings)

    # 질의응답 체인 생성
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAIEmbeddings().llm(),
        retriever=vector_store.as_retriever()
    )

    # 사용자 입력 받기
    query = st.text_input("질문을 입력하세요:")

    if query:
        # 답변 생성
        response = qa_chain.run(query)
        st.write("### 답변:")
        st.write(response)
