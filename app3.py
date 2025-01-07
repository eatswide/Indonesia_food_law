import streamlit as st
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# OpenAI 클라이언트 설정
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# 제목 설정
st.title("인도네시아 식품법 챗봇")

# 시스템 프롬프트 설정
SYSTEM_PROMPT = """
당신은 인도네시아 식품법 전문가입니다.
주어진 문서의 내용을 바탕으로 다음 지침에 따라 정확하고 신뢰성 있는 답변을 제공하세요:

1. 답변은 다음 형식을 따르세요:
   - **관련 법조항 요약**: 간결하고 핵심적인 문장으로 요약
   - **상세 설명**: 명확하고 구체적인 정보를 제공하며, 문서 내용과 연결
   - **실제 적용 방법이나 예시**: 현실적인 사례 또는 단계 포함
2. 항상 관련 법조항, 규정 번호 또는 제목을 구체적으로 인용하세요.
3. 불확실한 부분이 있다면 명확히 언급하고, 추가 자료를 참조하라고 안내하세요.
4. 가능한 경우 구체적인 수치, 기준, 또는 법적 요구 사항을 포함하여 답변의 신뢰성을 높이세요.
5. 전문 용어는 풀어서 설명하되, 필요할 경우 원어(인도네시아어)를 괄호 안에 병기하세요.
6. 사용자의 질문이 모호할 경우, 필요한 추가 정보를 요청하세요.
7. 답변이 끝난 후, "이와 관련된 다른 질문이 있으시면 말씀해주세요."라고 마무리하세요.

문서 내용:
{context}

질문:
{question}
"""


# 초기 설정
@st.cache_resource
def load_data():
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
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
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT.format(context=context, question=user_question)},
                {"role": "user", "content": user_question}
            ],
            temperature=0.7,  # 답변의 창의성 조절 (0: 보수적, 1: 창의적)
            max_tokens=1000   # 답변 길이 제한
        )
        
        # 답변 표시
        st.write("답변:", response.choices[0].message.content)

except Exception as e:
    st.error(f"데이터 로딩 중 오류가 발생했습니다: {str(e)}")