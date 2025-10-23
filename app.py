# app.py - Final version with forced error reporting
import warnings
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from dotenv import load_dotenv
import os
from functools import wraps
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
import datetime # Import for logging timestamps

# --- Suppress Warnings ---
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress TensorFlow warnings

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

docsearch = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings
)

# --- UPDATED MODEL SELECTION ---
MODEL_NAME = "nvidia/nemotron-nano-9b-v2"

# Initialize the LLM to use OpenRouter with the selected model
llm = ChatOpenAI(
    model=MODEL_NAME,
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.1
)

retriever = docsearch.as_retriever(search_kwargs={"k": 3})

prompt_template = """
You are a helpful customer support assistant. Use the following pieces of context to answer the question at the end.

Context: {context}

Question: {question}

Instructions:
1. If the context provides a clear answer to the question, provide that "Helpful Answer".
2. If the user's question is very broad or a single keyword (e.g., "invoices", "shipping", "returns"), do not just summarize the context. Instead, first acknowledge the topic and then ask a clarifying question or provide a list of more specific questions you can answer based on the context.
3. If the context does not contain the answer, just say that you don't have information on that topic. Do not try to make up an answer.

Helpful Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

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

@app.route("/ask", methods=["POST"])
@login_required
def get_bot_response():
    print("--- A new message was received ---")
    
    if request.is_json:
        data = request.get_json()
        msg = data.get("question", "")
    else:
        msg = request.form.get("msg", "")
    
    print(f"User Input: {msg}")
    
    if not msg:
        return jsonify({"response": "Please provide a question."})
    
    print("Invoking the chain...")
    try:
        response = chain.invoke(msg)
        print("Chain executed successfully.")
        
        answer = response["result"]
        print("Final Answer: ", answer)
        
        return jsonify({"response": answer})
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"response": "An error occurred while processing your request."})

# --- NEW ROUTE FOR LOGGING FEEDBACK ---
@app.route("/log_feedback", methods=["POST"])
@login_required
def log_feedback():
    try:
        data = request.get_json()
        message = data.get("message", "")
        feedback = data.get("feedback", "")
        
        if not message or not feedback:
            return jsonify({"status": "error", "message": "Missing data"}), 400

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] Feedback: {feedback} | Message: {message}\n"
        
        with open("feedback.log", "a") as f:
            f.write(log_entry)
            
        return jsonify({"status": "success", "message": "Feedback logged"})
    except Exception as e:
        print(f"Error logging feedback: {str(e)}")
        return jsonify({"status": "error", "message": "Failed to log feedback"}), 500

if __name__ == '__main__':
    app.run(debug=True)