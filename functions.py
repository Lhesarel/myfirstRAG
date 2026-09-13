import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def load_and_split_documents(data_dir, chunk_size=500, chunk_overlap=100):
    # Load all PDF files from the 'data' directory and its subdirectories
    loader = DirectoryLoader(data_dir, glob='**/*.pdf', loader_cls=PyPDFLoader)
    pages = loader.load()

    # Split the loaded pages into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(pages)

    return chunks

def create_db_and_retriever(chunks, k=4):
    # Generate embeddings for the chunks, store them in a vector database and create a retriever
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if os.path.exists("chroma_db"):
        db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    else:
        db = Chroma.from_documents(chunks, embeddings, persist_directory="chroma_db")
    retriever = db.as_retriever(search_kwargs={"k": k})
    
    return retriever

def join_text(docs):
    # Join the text content of the retrieved documents into a single string
    return '\n\n'.join([doc.page_content for doc in docs])