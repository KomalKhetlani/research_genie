import os
import chromadb
import ollama
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
CHROMA_DB_LLAMA3_FOLDER_PATH = os.getenv('CHROMA_DB_LLAMA3_FOLDER_PATH')
# Load the ChromaDB persistent database
client = chromadb.PersistentClient(path=CHROMA_DB_LLAMA3_FOLDER_PATH)

# Load the collection where embeddings are stored
collection = client.get_collection(name="research_chunks")


def get_embedding(text):
    """Generate embeddings using LLaMA 3 with Ollama."""
    response = ollama.embeddings(model="llama3", prompt=text)
    return response["embedding"]


def retrieve_relevant_chunks(query, top_k=5):
    """Retrieve the top-k most relevant document chunks for a given query."""

    # Convert user query to embedding using LLaMA 3 + Ollama
    query_embedding = get_embedding(query)

    # Search for top-k similar embeddings in ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k  # Get top-k relevant chunks
    )

    # Extract the relevant text chunks
    retrieved_chunks = results["documents"][0]  # List of retrieved text chunks
    retrieved_metadata = results["metadatas"][0]  # Metadata (e.g., source document)

    return retrieved_chunks, retrieved_metadata


def generate_answer(query):
    """Retrieve relevant chunks and use LLaMA 3 to generate an answer."""

    # Retrieve relevant chunks
    retrieved_chunks, _ = retrieve_relevant_chunks(query)

    # Format retrieved chunks as context
    context = "\n\n".join(retrieved_chunks)

    # Create the final prompt for LLaMA 3
    prompt = f"""You are an AI assistant. Use the following retrieved information to answer the user's query:

        Context:
        {context}

        Query: {query}

        Provide a well-structured answer based on the provided context.
        """

    # Generate an answer using LLaMA 3
    response = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])

    return response["message"]["content"]


# Example Query
query = "What is Large Language Model?"
answer = generate_answer(query)

# Print the Answer
print("\n📝 Generated Answer:\n")
print(answer)

