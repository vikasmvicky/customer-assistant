# app.py - Final version with forced error reporting

from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from dotenv import load_dotenv
import os
from functools import wraps
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer

load_dotenv()

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['SECRET_KEY'] = 'a-very-secret-and-long-string-that-you-should-change'

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
INDEX_NAME = os.environ.get("INDEX_NAME", "customassistant")

VALID_USERNAME = "admin"
VALID_PASSWORD = "password"

# --- Initialize Clients and Chains ---
print("Loading embedding model...")
class LocalEmbeddings:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()
    def embed_query(self, text):
        return self.model.encode(text).tolist()

embeddings = LocalEmbeddings('sentence-transformers/all-MiniLM-L6-v2')
print("Model loaded.")

# Connect to Pinecone using LangChain's wrapper
docsearch = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings
)

# Initialize the LLM to use OpenRouter
llm = ChatOpenAI(
    model="meta-llama/llama-3.1-8b-instruct:free",
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.1
)

# Create the retriever
retriever = docsearch.as_retriever(search_kwargs={"k": 3})

# Define the prompt template
prompt_template = """
Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Helpful Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# Create the RetrievalQA chain
chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

# --- Login and Routes ---
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            flash("You need to be logged in to access this page.")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    if 'user' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == VALID_USERNAME and password == VALID_PASSWORD:
            session['user'] = username
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid username or password.")
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', title='Customer Assistant')

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash("You have been logged out.")
    return redirect(url_for('login'))

@app.route("/get", methods=["POST"])
@login_required
def get_bot_response():
    print("--- A new message was received ---")
    
    msg = request.form["msg"]
    print(f"User Input: {msg}")
    
    # Use the QA chain to get the answer
    print("Invoking the chain...")
    response = chain.invoke(msg)
    print("Chain executed successfully.")
    
    answer = response["result"]
    print("Final Answer: ", answer)
    
    return jsonify({"response": answer})

if __name__ == '__main__':
    app.run(debug=True)