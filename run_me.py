# run_me.py
import os
import time
import warnings
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from sentence_transformers import SentenceTransformer
# --- NEW IMPORTS FOR OCR ---
import pytesseract
from PIL import Image

# --- Suppress Warnings ---
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress TensorFlow warnings

load_dotenv()

# --- Configuration ---
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
INDEX_NAME = os.environ.get("INDEX_NAME", "customassistant")
DIMENSION = 384
DATA_DIRECTORY = "data"

# --- Initialize Clients ---
print("Loading embedding model... (This may take a minute the first time)")

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

# --- Helper Functions ---
def check_data_directory():
    if not os.path.exists(DATA_DIRECTORY):
        print(f"Error: Directory '{DATA_DIRECTORY}' does not exist.")
        print("Please create it and add your PDF files.")
        return False
    
    pdf_files = [os.path.join(DATA_DIRECTORY, f) for f in os.listdir(DATA_DIRECTORY) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"Error: No PDF files found in '{DATA_DIRECTORY}' directory.")
        return False
    
    return True

# --- UPDATED FUNCTION with OCR Fallback ---
def extract_text_from_pdfs(pdf_files):
    """Extract text from PDFs, using OCR as a fallback for scanned pages."""
    raw_text = ""
    for pdf_path in pdf_files:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                print(f"Processing {os.path.basename(pdf_path)}...")
                for i, page in enumerate(pdf.pages):
                    # First, try to extract text directly
                    page_text = page.extract_text()
                    
                    if page_text and page_text.strip():
                        raw_text += page_text + "\n"
                        print(f"  - Page {i+1}: Text extracted directly.")
                    else:
                        # If no text, try OCR
                        print(f"  - Page {i+1}: No text found, trying OCR...")
                        try:
                            # Convert the PDF page to an image for OCR
                            img = page.to_image(resolution=300).original
                            ocr_text = pytesseract.image_to_string(img)
                            if ocr_text.strip():
                                raw_text += ocr_text + "\n"
                                print(f"  - Page {i+1}: OCR successful.")
                            else:
                                print(f"  - Page {i+1}: OCR found no text.")
                        except Exception as ocr_error:
                            print(f"  - Page {i+1}: OCR failed. Error: {ocr_error}")
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            continue
    
    # --- DEBUG: Print the first 500 characters of the extracted text ---
    print("\n--- DEBUG: First 500 characters of extracted text ---")
    print(raw_text[:500])
    print("-----------------------------------------------------\n")
    
    return raw_text

def confirm_index_deletion():
    response = input(f"Are you sure you want to delete the existing index '{INDEX_NAME}'? (y/n): ")
    return response.lower() == 'y'

# --- Main Logic ---
if __name__ == '__main__':
    print("Starting data ingestion process...")
    
    if not check_data_directory():
        exit()
    
    pdf_files = [os.path.join(DATA_DIRECTORY, f) for f in os.listdir(DATA_DIRECTORY) if f.endswith(".pdf")]
    print(f"Found {len(pdf_files)} PDF files.")
    
    if INDEX_NAME in pc.list_indexes().names():
        if not confirm_index_deletion():
            print("Index deletion cancelled. Exiting.")
            exit()
            
        print(f"Deleting existing index '{INDEX_NAME}'...")
        try:
            pc.delete_index(INDEX_NAME)
            print("Waiting for index deletion to complete...")
            time.sleep(10)
        except Exception as e:
            print(f"Error deleting index: {str(e)}")
            exit()

    print(f"Creating new index '{INDEX_NAME}'...")
    try:
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print("Waiting for index to be ready...")
        time.sleep(10)
        print("Index created successfully.")
    except Exception as e:
        print(f"Error creating index: {str(e)}")
        exit()

    print("Extracting text from PDFs...")
    raw_text = extract_text_from_pdfs(pdf_files)
    
    if not raw_text.strip():
        print("No text could be extracted from the PDFs. Exiting.")
        exit()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text_chunks = text_splitter.split_text(raw_text)
    print(f"Split text into {len(text_chunks)} chunks.")

    print("Upserting embeddings to Pinecone using LangChain...")
    try:
        PineconeVectorStore.from_texts(
            texts=text_chunks,
            embedding=embeddings,
            index_name=INDEX_NAME
        )
        print("Data ingestion complete!")
    except Exception as e:
        print(f"Error during data ingestion: {str(e)}")
        exit()