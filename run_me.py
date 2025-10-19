# run_me.py
import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from sentence_transformers import SentenceTransformer

load_dotenv()

# --- Configuration ---
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
INDEX_NAME = os.environ.get("INDEX_NAME", "customassistant")
DIMENSION = 384 # Dimension for 'all-MiniLM-L6-v2'

# --- Initialize Clients ---
print("Loading embedding model... (This may take a minute the first time)")
# Use a wrapper class to make sentence-transformers compatible with LangChain
class LocalEmbeddings:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()
    def embed_query(self, text):
        return self.model.encode(text).tolist()

embeddings = LocalEmbeddings('sentence-transformers/all-MiniLM-L6-v2')
print("Model loaded.")

pc = Pinecone(api_key=PINECONE_API_KEY)

# --- Main Logic ---
if __name__ == '__main__':
    print("Starting data ingestion process...")

    # Delete old index if it exists
    if INDEX_NAME in pc.list_indexes().names():
        print(f"Deleting existing index '{INDEX_NAME}'...")
        pc.delete_index(INDEX_NAME)
        time.sleep(10)

    # Create new index
    print(f"Creating new index '{INDEX_NAME}'...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    time.sleep(5)
    print("Index created successfully.")

    # Load and process PDFs
    pdf_directory = "data"
    pdf_files = [os.path.join(pdf_directory, f) for f in os.listdir(pdf_directory) if f.endswith(".pdf")]
    
    if not pdf_files:
        print("No PDF files found. Exiting.")
        exit()

    print(f"Found {len(pdf_files)} PDF files. Processing...")
    raw_text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            raw_text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text_chunks = text_splitter.split_text(raw_text)
    print(f"Split text into {len(text_chunks)} chunks.")

    # Use LangChain's PineconeVectorStore to ingest data
    print("Upserting embeddings to Pinecone using LangChain...")
    PineconeVectorStore.from_texts(
        texts=text_chunks,
        embedding=embeddings,
        index_name=INDEX_NAME
    )
    print("Data ingestion complete!")