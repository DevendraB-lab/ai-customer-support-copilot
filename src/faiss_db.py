import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load customer support FAQs
with open("data/customer_support_docs.json", "r", encoding="utf-8") as file:
    documents = json.load(file)

# Load a sentence embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert questions into embeddings
questions = [doc["question"] for doc in documents]
question_embeddings = model.encode(questions)

# Convert to NumPy array
embeddings_array = np.array(question_embeddings, dtype="float32")

# Create FAISS index
index = faiss.IndexFlatL2(embeddings_array.shape[1])  # L2 distance
index.add(embeddings_array)

# Save FAISS index
faiss.write_index(index, "data/faiss_index.idx")

print("FAISS index created and saved!")
