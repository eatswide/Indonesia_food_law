import streamlit as st
import PyPDF2
import openai
import os

# OpenAI API 키 설정
# openai.api_key = "당신의_OPENAI_API_KEY" 이 부분을 아래처럼 변경
openai.api_key = st.secrets["OPENAI_API_KEY"]

# 제목 설정
st.title("PDF 문서 검색 챗봇")

# PDF 파일 업로드 위젯
uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type="pdf")

if uploaded_file is not None:
    # PDF 텍스트 추출
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # 사용자 입력 받기
    user_question = st.text_input("질문을 입력하세요:")

    if user_question:
        # OpenAI API를 사용하여 답변 생성
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"다음 문서를 바탕으로 질문에 답변해주세요: {text}"},
                {"role": "user", "content": user_question}
            ]
        )
        
        # 답변 표시
        st.write("답변:", response.choices[0].message.content)