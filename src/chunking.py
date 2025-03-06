import os
import json
import PyPDF2
import ollama
import re
import tiktoken
from dotenv import load_dotenv
load_dotenv()
RESEARCH_PAPER_PDF_FOLDER_PATH = os.getenv('RESEARCH_PAPER_PDF_FOLDER_PATH')
RESEARCH_PAPER_JSON_FOLDER_PATH = os.getenv('RESEARCH_PAPER_JSON_FOLDER_PATH')
RESEARCH_PAPER_CHUNKS_FOLDER_PATH = os.getenv('RESEARCH_PAPER_CHUNKS_FOLDER_PATH')

def extract_text_from_pdfs(pdf_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            json_filename = os.path.splitext(filename)[0] + ".json"
            json_path = os.path.join(output_folder, json_filename)

            try:
                with open(pdf_path, "rb") as pdf_file:
                    reader = PyPDF2.PdfReader(pdf_file)
                    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

                if text.strip():
                    with open(json_path, "w", encoding="utf-8") as json_file:
                        json.dump({"filename": filename, "text": text}, json_file, ensure_ascii=False, indent=4)

                    print(f"Extracted text saved to {json_path}")
                else:
                    print(f"Skipping {filename} as no text was extracted.")
            except Exception as e:
                print(f"Error processing {filename}: {e}")



def sliding_window_chunking(text, window_size=512, overlap=128):
        """
        Splits text using a sliding window approach with overlap, ensuring full sentences.

        Args:
            text (str): Input text to be chunked.
            window_size (int): Number of tokens per chunk.
            overlap (int): Number of overlapping tokens.

        Returns:
            list: List of chunked text segments.
        """
        enc = tiktoken.get_encoding("cl100k_base")  # Use OpenAI's tokenizer
        tokens = enc.encode(text, disallowed_special=())  # Convert text into tokens
        decoded_text = enc.decode(tokens)  # Convert back to text

        # Split text into sentences using regex (or use spaCy for better accuracy)
        sentences = re.split(r'(?<=[.!?])\s+', decoded_text)  # Split at sentence boundaries

        chunks = []
        current_chunk = []
        current_length = 0
        chunk_id = 1

        for sentence in sentences:
            sentence_tokens = enc.encode(sentence, disallowed_special=())
            sentence_length = len(sentence_tokens)

            if current_length + sentence_length > window_size:
                # If adding a sentence exceeds window size, finalize the current chunk
                chunks.append({"chunk_id": chunk_id, "text": enc.decode(sum(current_chunk, []))})
                chunk_id += 1

                # Start a new chunk with overlap from the previous chunk
                current_chunk = current_chunk[-overlap:] if overlap > 0 else []
                current_length = sum(len(t) for t in current_chunk)

            # Add the current sentence to the chunk
            current_chunk.append(sentence_tokens)
            current_length += sentence_length

        # Add the last chunk
        if current_chunk:
            chunks.append({"chunk_id": chunk_id, "text": enc.decode(sum(current_chunk, []))})

        return chunks



def process_json_files(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            print('Input Path = ', input_path)
            with open(input_path, "r", encoding="utf-8") as file:
                data = json.load(file)
                text = data.get("text", "")
                if text:
                    chunks = sliding_window_chunking(text)

                    chunked_data = {"filename": filename, "chunks": chunks}

                    with open(output_path, "w", encoding="utf-8") as out_file:
                        json.dump(chunked_data, out_file, ensure_ascii=False, indent=4)

                else:
                    print('No text found')



def main():
    pdf_folder = RESEARCH_PAPER_PDF_FOLDER_PATH
    output_folder = RESEARCH_PAPER_JSON_FOLDER_PATH
    chunked_output_folder = RESEARCH_PAPER_CHUNKS_FOLDER_PATH
    if not os.listdir(output_folder):
        extract_text_from_pdfs(pdf_folder, output_folder)
    process_json_files(output_folder, chunked_output_folder)


if __name__ == '__main__':
    main()