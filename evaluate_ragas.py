from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas import evaluate
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig
from langchain_ollama import ChatOllama,OllamaEmbeddings
import pandas as pd
import time


PDF_PATH = "./Attention_Is_All_You_Need.pdf"


def build_pipeline(pdf_path) :
    docs = PyPDFLoader(pdf_path).load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    vector_db = Chroma.from_documents(documents=chunks, embedding=embeddings, collection_name="eval_run")
    vector_retriever = vector_db.as_retriever(search_kwargs={"k": 10})

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 10

    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever], weights=[0.5, 0.5])

    cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=6)
    retriever = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=ensemble_retriever)

    chat_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    template = """You are a strict question answering assistant. 
    Answer the query ONLY using information explicitly stated in the provided context. 
    Do NOT use any prior knowledge or make any inferences beyond what is written in the context.
    If the exact answer is not present in the context, respond with "The context does not contain enough information to answer this question. 
    context : {context} 
    query : {query} """
    prompt = PromptTemplate.from_template(template)

    return retriever, chat_llm, prompt


def format_docs(docs) :
    return "\n\n".join(d.page_content for d in docs)


def main() :
    retriever, chat_llm, prompt = build_pipeline(PDF_PATH)
    answer_chain = prompt | chat_llm | StrOutputParser()
    

    golden_df = pd.read_csv("golden_dataset.csv")

    samples = []

    for _, row in golden_df.iterrows():

        question = row["user_input"]
        reference = row["reference"]

        retrieved_docs = retriever.invoke(question)

        contexts = [
            doc.page_content
            for doc in retrieved_docs
        ]

        answer = answer_chain.invoke(
            {
                "context": format_docs(retrieved_docs),
                "query": question,
            }
        )

        samples.append(
            SingleTurnSample(
                user_input=question,
                retrieved_contexts=contexts,
                response=answer,
                reference=reference,
            )
        )
        time.sleep(10)

    dataset = EvaluationDataset(samples=samples)

    judge_llm = LangchainLLMWrapper(ChatOllama(model="mistral-gpu:latest", temperature=0))
    judge_embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="nomic-embed-text"))
    
    metrics = [
        Faithfulness(),
        AnswerRelevancy(),
        ContextPrecision(),
        ContextRecall(),
    ]

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=RunConfig(timeout=300, max_retries=10, max_workers=1)
    )

    print(result)
    result.to_pandas().to_csv("ragas_results.csv", index=False)


if __name__ == "__main__" :
    main()