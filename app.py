from flask import Flask, jsonify, request
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import chromadb
import requests
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Debug: Print environment variables (without exposing full API key)
print(f"OPENAI_BASE_URL: {OPENAI_BASE_URL}")
print(f"OPENAI_API_KEY: {'*' * 20 + (OPENAI_API_KEY[-4:] if OPENAI_API_KEY else 'NOT SET')}")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# # Load the LLaMA model and tokenizer
# llama_model_name = "meta-llama/Llama-2-7b-hf" 

# try:
#     bnb_config = BitsAndBytesConfig(load_in_4bit=True)
#     tokenizer = AutoTokenizer.from_pretrained(llama_model_name)
#     llama_model = AutoModelForCausalLM.from_pretrained(
#         llama_model_name,
#         low_cpu_mem_usage=True,
#         device_map="auto",  # Automatically map to GPU/CPU
#         torch_dtype=torch.float32
#     )
#     llama_model.eval()  # Set the model to evaluation mode
# except Exception as e:
#     print(f"Error loading LLaMA model: {e}")
#     exit(1)

# # Check if CUDA is available
# device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Sentence Transformer for query vectorization
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize ChromaDB client
try:
    client = chromadb.PersistentClient(path="chroma_db/")
    collection = client.get_collection('far_embedded_data')
    print("Successfully connected to FAR embedded data collection")
except Exception as e:
    print(f"Error initializing ChromaDB: {e}")
    exit(1)

@app.route('/query', methods=['POST'])
def handle_query():
    user_query = request.json.get('query', '')
    if not user_query:
        return jsonify({"error": "Invalid input. Query is required."}), 400
    
    try:
        response = process_query(user_query)
        return jsonify({"response": response})
    except Exception as e:
        print(f"Error processing query: {e}")
        return jsonify({"error": "An error occurred while processing the query."}), 500

def vectorize_query(user_query):
    """Generate a vector representation of the user's query."""
    return embedding_model.encode(user_query)

def retrieve_similar_docs(query_vector, top_k=1):
    """Retrieve similar documents from ChromaDB."""
    try:
        results = collection.query(
            query_embeddings=[query_vector],  # Pass as a list
            n_results=top_k
        )
        return results
    except Exception as e:
        print(f"Error retrieving documents from ChromaDB: {e}")
        return {"documents": []}

def format_context(retrieved_documents):
    """Format the retrieved documents into a context string."""
    if not retrieved_documents or "documents" not in retrieved_documents:
        return "No relevant FAR information found in the database."
    flat_documents = [
        doc[0] if isinstance(doc, list) and len(doc) > 0 else ""
        for doc in retrieved_documents.get('documents', [])
    ]
    return "\n".join(flat_documents)

def query_llm_with_llama(user_query, context):
    """Generate a response using the external API."""
    if not OPENAI_API_KEY or not OPENAI_BASE_URL:
        print("Error: OPENAI_API_KEY or OPENAI_BASE_URL not configured")
        return "Error: API configuration missing."
    
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "gpt-3.5-turbo",  # Add model specification
        "messages": [
            {"role": "system", "content": "You are a helpful assistant specializing in Federal Acquisition Regulation (FAR) information. You have access to a comprehensive database of FAR regulations and procedures. Only provide information that is available in the FAR database. If asked about topics not covered in the FAR or if the specific information is not available in the database, clearly state that the information is not available in the FAR database. When answering questions, cite relevant FAR sections when possible and provide accurate, detailed information based solely on the FAR content provided in the context."},
            {"role": "system", "content": f"FAR Context: {context}"},
            {"role": "user", "content": user_query}
        ],
        "max_tokens": 500,
        "temperature": 0.7
    }
    
    try:
        # Fix URL formatting - ensure it ends with /
        api_url = OPENAI_BASE_URL
        if not api_url.endswith('/'):
            api_url += '/'
        api_url += 'chat/completions'
        
        response = requests.post(
            url=api_url,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        print(f"API Response Status: {response.status_code}")
        if response.status_code != 200:
            print(f"API Response: {response.text}")
        
        response.raise_for_status()
        data = response.json()
        
        # Extract the model's reply
        if "choices" in data and len(data["choices"]) > 0:
            reply = data["choices"][0].get("message", {}).get("content", "No response.")
            return reply.strip()
        else:
            return "No response received from API."
            
    except requests.exceptions.RequestException as e:
        print(f"Error querying external API: {e}")
        return "Error: Unable to generate a response."
    except Exception as e:
        print(f"Unexpected error: {e}")
        return "Error: Unexpected error occurred."
def process_query(user_query):
    """Process the user query."""
    # Step 1: Vectorize the query
    query_vector = vectorize_query(user_query)
    
    # Step 2: Retrieve similar documents
    retrieved_documents = retrieve_similar_docs(query_vector)
    
    # Step 3: Format the context
    context = format_context(retrieved_documents)
    
    # Step 4: Query the LLaMA model
    response = query_llm_with_llama(user_query, context)
    
    return response

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
