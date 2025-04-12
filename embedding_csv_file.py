import pandas as pd
from chromadb.config import Settings
import chromadb
from sentence_transformers import SentenceTransformer

# Load CSV data
csv_file_path = "product_details.csv"
df = pd.read_csv(csv_file_path)

# Initialize ChromaDB client
client = chromadb.PersistentClient(path="chroma_db/")

# Initialize embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Define collection name
collection_name = "laptop_embedded_data"

# Check if collection exists, if so, delete it
existing_collections = [col for col in client.list_collections()]
if collection_name in existing_collections:
    client.delete_collection(collection_name)

# Create new collection
collection = client.create_collection(collection_name)

# Add documents to ChromaDB
for idx, row in df.iterrows():
    content = f"{row['title']} - {row['short_description']} - {row['description']}"
    embedding = model.encode(content).tolist()  # Generate embedding

    collection.add(
        documents=[content],
        embeddings=[embedding],
        metadatas=[{
            "id": str(row["id"]),
            "title": row["title"],
            "brand_id": row["brand_id"],
            "rating": row.get("expert_rating", None),
            "release_date": row.get("release_date", None)
        }],
        ids=[str(row["id"])]
    )

print("CSV data embedded and stored in ChromaDB.")
