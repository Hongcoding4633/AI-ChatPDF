# 클라우드용
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# PC 로컬용
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
import chromadb
import streamlit as st
import tempfile
import os
from streamlit_extras.buy_me_a_coffee import button
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
import chromadb.api
chromadb.api.client.SharedSystemClient.clear_system_cache()

button(username="nofearhong", floating=True, width=221)

# 제목
st.title("ChatPDF")
st.write("---")

# API 키 확인
openai_key = st.text_input("OPENAI_API_KEY", type="password")

# 파일 업로드
uploaded_file = st.file_uploader("PDF 파일을 선택하세요.", type=['pdf'])
st.write("---")

def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

# 업로드되면 동작하는 코드
if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)

    # Split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, 
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    texts = text_splitter.split_documents(pages)

    # Embedding
    embeddings_model = OpenAIEmbeddings(openai_api_key=openai_key)

    # ChromaDB 클라이언트 생성
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    
    # Chroma 데이터베이스 생성
    db = Chroma.from_documents(
        documents=texts,
        embedding=embeddings_model)
    
    # Stream 받아줄 Handler 만들기
    class StreamHandler(BaseCallbackHandler):
        def __init__(self, container, initial_text=""):
            self.container = container
            self.text = initial_text
        def on_llm_new_token(self, token: str, **kwargs) -> None:
            self.text+=token
            self.container.markdown(self.text)

    # Question
    st.header("PDF에게 질문해보라!")
    question = st.text_input("질문을 입력하세요.")  
    
    if st.button("질문하기"):
        with st.spinner("Waiting for..."):
            chat_box = st.empty()
            stream_handler = StreamHandler(chat_box)
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", 
                            temperature=0, 
                            openai_api_key=openai_key,
                            streaming=True,
                            callbacks=[stream_handler]
                            )
            qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
            qa_chain({"query": question})