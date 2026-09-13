import os
import sys
import functions as F
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def main():
    chunk_size = int(os.environ.get("CHUNK_SIZE", 500))
    chunk_overlap = int(os.environ.get("CHUNK_OVERLAP", 100))
    k = int(os.environ.get("K", 4))
    temperature = float(os.environ.get("TEMPERATURE", 0))
    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")

    # Load and split documents from the 'data' directory
    try:
        chunks = F.load_and_split_documents(data_dir="data", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    except Exception as e:
        print(f"Error loading documents: {e}")
        sys.exit(1)
        
    # Create a vector database and retriever from the chunks
    retriever = F.create_db_and_retriever(chunks, k=k)

    # Instantiate the LLM
    llm = ChatOllama(model="qwen3:8b", temperature=temperature, base_url=ollama_url) # Temperature is set to 0 for deterministic responses
    
    # Build the prompt template for the RAG chain
    prompt = ChatPromptTemplate.from_template(
        "Answer the question based on the context below.\n"
        "If the answer is not contained within the context, respond with 'I don't know'.\n\n" # To avoid hallucinations
        "Context: \n{context}\n\n"
        "Question: \n{question}\n\n"
        "Answer:"
    )    

    # Build the RAG chain by combining the retriever, prompt, and LLM
    chain = (prompt | llm | StrOutputParser())
    
    # The main loop to interact with the user
    print("Welcome to the RAG system! Type your question or 'exit' to quit.\n")
    user_input = input()
    while user_input != "exit":
        # Answer the question using the RAG chain
        docs = retriever.invoke(user_input)
        context = F.join_text(docs)
        response = chain.invoke({"context": context, "question": user_input})
        sources = sorted({f"{d.metadata.get('source')} p.{d.metadata.get('page')+1}" for d in docs})
            
        print(response)
        if not response.strip().lower().startswith("i don't know"): 
            print(f"\nSources: {', '.join(map(str, sources))}")
        print("-"*50)
        print("\nWhat more do you want to know?")
        user_input = input()


if __name__ == "__main__":
    main()