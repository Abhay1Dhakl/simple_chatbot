from flask import Flask, jsonify, request
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
app = Flask(__name__)

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
    collection = client.get_collection('laptop_embedded_data')
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
        return "No relevant documents found."
    flat_documents = [
        doc[0] if isinstance(doc, list) and len(doc) > 0 else ""
        for doc in retrieved_documents.get('documents', [])
    ]
    return "\n".join(flat_documents)

def query_llm_with_llama(user_query, context):
     """Generate a response using the external API."""
     headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
     payload = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer questions.Only answer about the gadgets that are available in out vector database and if you are asked about other question that are related to gadgets then dont answer it and also if that particular gadget is not available then replay we dont have that product here. If you are asked to list the product then provide all the products that matches with the question for example if you are asked to list the iphones then list all the available iphones that are in the vector storage."},
                {"role": "system", "content": f"Context: {context}"},
                {"role": "user", "content": user_query}
            ]
        }
        
     try:
            response = requests.post(
                url=f"{OPENAI_BASE_URL}chat/completions",  # Adjust endpoint as per API documentation
                headers=headers,
                json=payload
            )
            response.raise_for_status()  # Raise an exception for HTTP errors
            data = response.json()
            # Extract the model's reply
            reply = data.get("choices", [])[0].get("message", {}).get("content", "No response.")
            return reply.strip()
     except requests.exceptions.RequestException as e:
            print(f"Error querying external API: {e}")
            return "Error: Unable to generate a response."
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
