from dotenv import load_dotenv
load_dotenv()


from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from time import sleep


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


llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")


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
    vector_db = InMemoryVectorStore.from_documents(documents= docs, embedding= embeddings)


    st.session_state.vector_db = vector_db
    st.session_state.document_uploaded = True



with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712139.png", width=100)
    st.title("System Diagnostics")
    st.markdown("---")
    st.markdown("### Architecture")
    st.markdown("- **LLM Engine:** Gemini 2.5 Flash")
    st.markdown("- **Embeddings:** Gemini-Embedding-2")
    st.markdown("- **Vector Store:** In-Memory / LangChain")
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

        retriever = st.session_state.vector_db.as_retriever(search_type = "similarity" , search_kwargs = {"k" : 2})
        
        
        template = """You are a helpful assistant. Provide answer to the user's query based on the context provided and your existing knowledge. context : {context} query : {query} """
        prompt = PromptTemplate.from_template(template)

        def format_docs(docs) :
            context = ""
            for doc in docs :
                context += doc.page_content + "\n\n"
            return context
        
        rag_chain = ({"context" : retriever | format_docs, "query" : RunnablePassthrough()}
                     | prompt | llm | StrOutputParser())
        
        
        result = rag_chain.invoke(query)


        st.session_state.messages.append({"role" : "ai", "content" : result})
        st.chat_message("ai").markdown(result)
