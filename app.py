from dotenv import load_dotenv
load_dotenv()


from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from time import sleep
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
import os
import shutil
import uuid
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder


st.set_page_config(page_title="RAG-Powered Document Intelligence", page_icon="🧠", layout="wide")

st.markdown("""
<style>
    .stChatMessage {
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .st-emotion-cache-1c7y2kd {
        background-color: #f4f6f9;
    }
</style>
""", unsafe_allow_html=True)


llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash", temperature=0)


if "vector_db" not in st.session_state :
    st.session_state.vector_db = None

if "messages" not in st.session_state :
    st.session_state.messages = []


def document_process(path) :
    loader = PyPDFLoader(path)
    docs = loader.load()


    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    docs = splitter.split_documents(docs)


    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    CHROMA_DIR = "./chroma_db"
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
    collection_name = f"session_{uuid.uuid4().hex[:8]}"
    
    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=CHROMA_DIR
    )


    st.session_state.vector_db = vector_db
    st.session_state.doc_chunks = docs
    st.session_state.document_uploaded = True



with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712139.png", width=100)
    st.title("System Diagnostics")
    st.markdown("---")
    st.markdown("### Architecture")
    st.markdown("- **LLM Engine:** Gemini 2.5 Flash")
    st.markdown("- **Embeddings:** Gemini-Embedding-2")
    st.markdown("- **Vector Store:** ChromaDB")
    st.markdown("- **Retrieval:** Hybrid BM25 + Dense Semantic Search")
    st.markdown("- **Reranker:** Cross-Encoder")
    st.markdown("- **Memory:** Multi-turn Chat History")
    st.markdown("---")
    
    if st.button("Reset Conversation"):
        st.session_state.messages = []
        st.rerun()



st.subheader("RAG-Powered Document Intelligence")


if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False


if not st.session_state.document_uploaded :
    file = st.file_uploader(label = "Select your Document", type = "pdf")
    if file :
        with open("uploaded_document.pdf", "wb") as f :
            f.write(file.getvalue())
        with st.spinner("Processing...") :    
            document_process("./uploaded_document.pdf")
        st.markdown("Document Processed Successfully...")     
        sleep(2)
        st.rerun()   



if st.session_state.document_uploaded and st.session_state.vector_db :
    for oneMessage in st.session_state.messages :
        role = oneMessage["role"]
        content = oneMessage["content"]

        st.chat_message(role).markdown(content)


    query = st.chat_input("Ask anything...")
    if query :

        st.session_state.messages.append({"role" : "user", "content" : query})

        st.chat_message("user").markdown(query)

        vector_retriever = st.session_state.vector_db.as_retriever(search_type = "similarity" , search_kwargs = {"k" : 10})
        
        bm25_retriever = BM25Retriever.from_documents(st.session_state.doc_chunks)
        bm25_retriever.k = 10
        
        ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever , vector_retriever] , weights=[0.5, 0.5])
        cross_encoder = HuggingFaceCrossEncoder(model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2")
        reranker = CrossEncoderReranker(model=cross_encoder , top_n=6)
        
        retriever = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=ensemble_retriever)
        
        
        template = """You are a strict question answering assistant. 
        Answer the query ONLY using information explicitly stated in the provided context. 
        Do NOT use any prior knowledge or make any inferences beyond what is written in the context.
        If the exact answer is not present in the context, respond with "The context does not contain enough information to answer this question." 
        chat_history : {chat_history}
        context : {context} 
        query : {query}"""
        prompt = PromptTemplate.from_template(template)

        def format_docs(docs) :
            context = ""
            for doc in docs :
                context += doc.page_content + "\n\n"
            return context
        
        chat_history = "\n".join(
            f"{m['role'].upper()}:{m['content']}"
            for m in st.session_state.messages[:-1]
        )
        
        rag_chain = ({"context" : retriever | format_docs, "query" : RunnablePassthrough(), "chat_history" : lambda _ : chat_history}
                     | prompt | llm | StrOutputParser())
        
        
        result = rag_chain.invoke(query)


        st.session_state.messages.append({"role" : "ai", "content" : result})
        st.chat_message("ai").markdown(result)