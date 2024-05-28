import streamlit as st
import tiktoken
from loguru import logger

# ConversationalRetrievalChain: 문서에서 정보를 검색하고, 이를 기반으로 대화형 응답을 생성하는 체인을 설정하는 클래스
from langchain.chains import ConversationalRetrievalChain
# ChatOpenAI: OpenAI의 GPT 모델을 사용하여 채팅 응답을 생성하는 클래스
from langchain_community.chat_models import ChatOpenAI
# PyPDFLoader: PDF 파일을 로드하고 텍스트로 변환하는 클래스
from langchain_community.document_loaders import PyPDFLoader
# Docx2txtLoader: DOCX 파일을 로드하고 텍스트로 변환하는 클래스
from langchain_community.document_loaders import Docx2txtLoader
# UnstructuredPowerPointLoader: PPTX 파일을 로드하고 텍스트로 변환하는 클래스
from langchain_community.document_loaders import UnstructuredPowerPointLoader
# RecursiveCharacterTextSplitter: 긴 텍스트를 주어진 길이로 분할하는 클래스
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
# HuggingFaceEmbeddings: Hugging Face 모델을 사용하여 텍스트 임베딩을 생성하는 클래스
from langchain_community.embeddings import HuggingFaceEmbeddings
# ConversationBufferMemory: 대화 기록을 버퍼에 저장하는 클래스
from langchain.memory import ConversationBufferMemory
# FAISS: Facebook AI Similarity Search, 벡터 임베딩을 효율적으로 검색하는 라이브러리
from langchain_community.vectorstores import FAISS
# get_openai_callback: OpenAI API 호출 시 콜백을 처리하는 함수
from langchain_community.callbacks.manager import get_openai_callback
# StreamlitChatMessageHistory: Streamlit 채팅 메시지 기록을 관리하는 클래스
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
# 만약 웹사이트 크롤링을 이용한 질의응답시 필요한 라이브러리
from langchain_community.document_loaders import WebBaseLoader

def main():
    # 스트림릿 페이지 설정
    st.set_page_config(
    page_title="DirChat",
    page_icon=":books:")

    # 페이지 제목 설정
    st.title("_Private Data :red[QA Chat]_ :books:")

    # 세션 상태 초기화
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None
    
    # 사이드바를 통해 파일 업로드 및 OpenAI API 키 입력 받음
    with st.sidebar:
        uploaded_files = st.file_uploader("Upload your file", type=['pdf','docx','pptx'], accept_multiple_files=True)
        openurl        = st.text_input("Insert URL", key = "url",)
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        process = st.button("Process")
    
    # Process 버튼 클릭 시 실행
    if process:
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        
        # 파일에서 텍스트 추출
        if not openurl:
            files_text = get_text(uploaded_files)
            
            # 텍스트를 청크로 분할
            text_chunks = get_text_chunks(files_text)
            
            # 텍스트 청크를 벡터 스토어에 저장
            vectorstore = get_vectorstore(text_chunks)
        
        else:
            text_chunks = url_text_chunks(openurl)
            vectorstore = get_vectorstore(text_chunks)
            
        # 대화 체인 설정
        st.session_state.conversation = get_conversation_chain(vectorstore, openai_api_key)
        
        st.session_state.processComplete = True

    # 메시지 상태 초기화
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", 
                                        "content": "안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"}]

    # 채팅 메시지 출력
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 채팅 메시지 기록 관리
    history = StreamlitChatMessageHistory(key="chat_messages")

    # 사용자의 질문을 입력 받음
    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        # 어시스턴트의 응답 처리
        with st.chat_message("assistant"):
            chain = st.session_state.conversation

            with st.spinner("Thinking..."):
                result = chain({"question": query})
                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history']
                response = result['answer']
                source_documents = result['source_documents']

                st.markdown(response)
                with st.expander("참고 문서 확인"):
                    for doc in source_documents:
                        st.markdown(doc.metadata['source'] , help=doc.page_content)

        
        # 어시스턴트의 응답을 채팅 메시지에 추가
        st.session_state.messages.append({"role": "assistant", "content": response})

# 텍스트의 길이를 계산하는 함수
def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

# 파일에서 텍스트를 추출하는 함수
def get_text(docs):
    doc_list = []
    
    for doc in docs:
        file_name = doc.name  # doc 객체의 이름을 파일 이름으로 사용
        with open(file_name, "wb") as file:  # 파일을 doc.name으로 저장
            file.write(doc.getvalue())
            logger.info(f"Uploaded {file_name}")
        
        # 파일 형식에 따라 로더를 선택
        if '.pdf' in doc.name:
            loader = PyPDFLoader(file_name)
            documents = loader.load_and_split()
        elif '.docx' in doc.name:
            loader = Docx2txtLoader(file_name)
            documents = loader.load_and_split()
        elif '.pptx' in doc.name:
            loader = UnstructuredPowerPointLoader(file_name)
            documents = loader.load_and_split()

        doc_list.extend(documents)
    return doc_list

# 텍스트를 청크로 분할하는 함수
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=300,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks

def url_text_chunks(url):
    text_splitter = CharacterTextSplitter(        
        separator="\n\n",
        chunk_size=3000,     # 쪼개는 글자수
        chunk_overlap=300,   # 오버랩 글자수
        length_function=len,
        is_separator_regex=False,
    )
    return(WebBaseLoader(url).load_and_split(text_splitter))

# 텍스트 청크를 벡터 스토어에 저장하는 함수
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )  
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb

# 대화 체인을 설정하는 함수
def get_conversation_chain(vectorstore, openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name = 'gpt-3.5-turbo',temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        chain_type="stuff", 
        retriever=vectorstore.as_retriever(search_type = 'mmr', vervose = True), 
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose = True
    )
    return conversation_chain

# 메인 함수 실행
if __name__ == '__main__':
    main()