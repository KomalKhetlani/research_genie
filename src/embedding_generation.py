import ollama
import numpy as np
import json
import os
import chromadb
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
RESEARCH_PAPER_CHUNKS_FOLDER_PATH = os.getenv('RESEARCH_PAPER_CHUNKS_FOLDER_PATH')
BATCH_TRACKING_FILE = "processed_batches.json"  # File to track processed batches
CHROMA_DB_LLAMA3_FOLDER_PATH = os.getenv('CHROMA_DB_LLAMA3_FOLDER_PATH')

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(
    path=CHROMA_DB_LLAMA3_FOLDER_PATH)  # Persistent storage
collection = chroma_client.get_or_create_collection(name="research_chunks")


def load_chunks(json_folder):
    """Loads text chunks from JSON files and ensures unique chunk IDs."""
    chunks = []
    file_paths = [os.path.join(json_folder, file) for file in os.listdir(json_folder) if file.endswith(".json")]

    for file_path in file_paths:
        filename = os.path.splitext(os.path.basename(file_path))[0]  # Extract filename without extension

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

            if "chunks" in data and isinstance(data["chunks"], list):
                for chunk in data["chunks"]:
                    if "text" in chunk and "chunk_id" in chunk:  # Ensure keys exist
                        unique_id = f"{filename}_{chunk['chunk_id']}"  # Create unique ID
                        chunks.append({"chunk_id": unique_id, "text": chunk["text"]})

    return chunks


def generate_embedding(text, model="llama3"):
    """
    Generate text embeddings using LLaMA 3 with Ollama.

    Args:
        text (str): The input text to generate embeddings for.
        model (str): The LLaMA 3 model to use.

    Returns:
        list: The generated embedding as a list.
    """
    try:
        response = ollama.embeddings(model=model, prompt=text)
        return response.get("embedding", [])
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None


def store_embeddings_in_chroma(chunks):
    """Generates embeddings and stores them in ChromaDB."""
    for chunk in chunks:
        embedding = generate_embedding(chunk["text"])
        if embedding:
            collection.add(
                ids=[str(chunk["chunk_id"])],  # Unique ID for each chunk
                embeddings=[embedding],  # Store embedding
                documents=[chunk["text"]]  # Store text as metadata
            )
            print(f"Stored chunk {chunk['chunk_id']} in ChromaDB.")
        else:
            print(f"Failed to generate embedding for chunk {chunk['chunk_id']}.")


def load_last_processed_batch():
    """Loads the last processed batch index from file."""
    if os.path.exists(BATCH_TRACKING_FILE):
        with open(BATCH_TRACKING_FILE, "r") as f:
            return json.load(f).get("last_processed_batch", 0)
    return 0


def save_last_processed_batch(batch_index):
    """Saves the last processed batch index to file."""
    with open(BATCH_TRACKING_FILE, "w") as f:
        json.dump({"last_processed_batch": batch_index}, f)


def main(batch_size=10):
    """Processes text chunks in batches and automatically picks up from the last processed batch."""
    chunks = load_chunks(RESEARCH_PAPER_CHUNKS_FOLDER_PATH)
    if not chunks:
        print("No valid chunks found.")
        return

    print(f"Loaded {len(chunks)} chunks.")

    total_batches = len(chunks) // batch_size + (1 if len(chunks) % batch_size else 0)
    start_batch = load_last_processed_batch()

    for batch_num in range(start_batch, total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(chunks))
        batch_chunks = chunks[start_idx:end_idx]
        print(f"Processing batch {batch_num + 1}/{total_batches} ({start_idx}-{end_idx})...")
        store_embeddings_in_chroma(batch_chunks)
        save_last_processed_batch(batch_num + 1)
        print(f"Completed batch {batch_num + 1}/{total_batches}.")


if __name__ == '__main__':
    main()