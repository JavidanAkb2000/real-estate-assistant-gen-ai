from configparser import SectionProxy
from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path
import os

# Core LangChain community modules
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Text splitter (new subpackage)
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))

# Constants
CHUNK_SIZE = 1000
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"
COLLECTION_NAME = "real_estate"

llm = None
vector_store = None


def initialize_components():
    global llm, vector_store
    if llm is None:
        llm = ChatGroq(model='openai/gpt-oss-20b', temperature=0.9, max_tokens=500)

    if vector_store is None:
        ef = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'trust_remote_code': True}
        )

        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=ef,
            persist_directory=str(VECTORSTORE_DIR),
        )


def process_urls(urls):
    """Generator function that yields status updates"""
    global vector_store

    yield 'Initializing components...'

    initialize_components()

    vector_store.delete_collection()

    # Reinitialize after deletion
    ef = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'trust_remote_code': True}
    )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=ef,
        persist_directory=str(VECTORSTORE_DIR),
    )

    yield 'Loading data from URLs...'
    loader = UnstructuredURLLoader(
        urls=urls,
        headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    )
    data = loader.load()

    yield 'Splitting text into chunks...'
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        separators=['\n\n', '\n', '.', ' ']
    )
    docs = text_splitter.split_documents(data)

    yield 'Adding documents to vector database...'
    uuids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(docs, ids=uuids)

    yield f'Successfully processed {len(docs)} documents!'


def generate_answer(query):
    if not vector_store:
        raise RuntimeError('No vector database found')

    # Retrieve relevant documents
    retriever = vector_store.as_retriever()
    docs = retriever.invoke(query)

    # Format context from retrieved documents
    context = "\n\n".join([doc.page_content for doc in docs])
    sources = list(set([doc.metadata.get('source', '') for doc in docs]))

    # Create prompt
    prompt = ChatPromptTemplate.from_template(
        """Answer the question based on the following context:

Context: {context}

Question: {question}

Answer:"""
    )

    # Create chain
    chain = (
            {"context": lambda x: context, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    answer = chain.invoke(query)

    return answer, ", ".join(sources)


if __name__ == '__main__':
    urls = [
        "https://www.cnbc.com/2024/12/21/how-the-federal-reserves-rate-policy-affects-mortgages.html",
        "https://www.cnbc.com/2024/12/20/why-mortgage-rates-jumped-despite-fed-interest-rate-cut.html"
    ]

    for status in process_urls(urls):
        print(status)

    answer, sources = generate_answer('Tell me what was the 30 year fixed mortagate rate along with the date?')
    print(f"\nAnswer: {answer}")
    print(f"\nSources: {sources}")
