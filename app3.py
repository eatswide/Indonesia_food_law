import streamlit as st
import PyPDF2
import openai
import io

# OpenAI API 키 설정
openai.api_key = st.secrets["OPENAI_API_KEY"]

# 제목 설정
st.title("인도네시아 식품법 챗봇")

# PDF 파일 내용을 직접 코드에 포함
pdf_text = """여기에 PDF 내용을 직접 붙여넣기"""

# 사용자 입력 받기
user_question = st.text_input("질문을 입력하세요:")

if user_question:
    # OpenAI API를 사용하여 답변 생성
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"다음 문서를 바탕으로 질문에 답변해주세요: {pdf_text}"},
            {"role": "user", "content": user_question}
        ]
    )
    
    # 답변 표시
    st.write("답변:", response.choices[0].message.content)