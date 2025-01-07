import streamlit as st
import openai
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# OpenAI API 키 설정
openai.api_key = st.secrets["OPENAI_API_KEY"]

# 제목 설정
st.title("인도네시아 식품법 챗봇")

# 초기 설정
@st.cache_resource
def load_data():
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)  # 이 부분이 수정됨
    return vectorstore

try:
    vectorstore = load_data()
    # 사용자 입력 받기
    user_question = st.text_input("질문을 입력하세요:")

    if user_question:
        # 가장 관련 있는 문서 부분 검색
        docs = vectorstore.similarity_search(user_question, k=3)
        context = "\n".join([doc.page_content for doc in docs])
        
        # OpenAI API로 답변 생성
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"다음 문서를 바탕으로 질문에 답변해주세요: {context}"},
                {"role": "user", "content": user_question}
            ]
        )
        
        # 답변 표시
        st.write("답변:", response.choices[0].message.content)

except Exception as e:
    st.error(f"데이터 로딩 중 오류가 발생했습니다: {str(e)}")