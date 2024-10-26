from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

# Инициализация параметров и модели для повторного использования
def initialize_model_and_db():
    local_model = "mistral"
    llm = ChatOllama(model=local_model)
    return llm

def load_and_split_pdf(pdf_file):
    loader = UnstructuredPDFLoader(file_path=pdf_file)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    return chunks

def create_vector_db(chunks):
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=True),
        collection_name="local-rag",
    )
    return vector_db

def answer_question_from_pdf(vector_db, llm, question):
    query_prompt = PromptTemplate(
        input_variables=['question'],
        template="""You are an intelligent model designed to help the user answer questions. Your task is to provide answers to questions based only on the information contained in the uploaded PDF document. You must answer in German. If there is no information in the document, report that there is no answer. Original question: {question}. After answering to question. Forget all information and data from PDF."""
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(),
        llm,
        prompt=query_prompt,
    )
    template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain.invoke(input=question)
