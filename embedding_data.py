import json
from chromadb.config import Settings
import chromadb
from sentence_transformers import SentenceTransformer

# initializing chromadb client
client = chromadb.PersistentClient(path="chroma_db/")

# load FAR data
with open("processed_far.json", "r", encoding="utf-8") as f:
    far_data = json.load(f)

# initialize embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# check if collection exists, if not create it
collection_name = "far_embedded_data"
print("hello",client.list_collections())
for name in client.list_collections():
    print("names aare",name.name)
    if collection_name == name.name:
        print("matched")
        client.delete_collection(collection_name)  # retrieve the existing collection
    
collection = client.create_collection(collection_name)  # create a new collection

# add documents to chromadb
for idx, chunk in enumerate(far_data):
    embedding = model.encode(chunk["content"]).tolist()  # generate embedding
    collection.add(
        documents=[chunk["content"]],
        embeddings=[embedding],
        metadatas=[{"page": chunk["page_number"], "section_number": chunk["section_number"], "section_title": chunk["metadata"]["section_title"]}],
        ids=[f"doc_{idx}"]
    )
    

print("FAR data embedded and stored in ChromaDB.")
