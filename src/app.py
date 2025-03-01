from fastapi import FastAPI
import openai
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel

# Load OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load FAISS index and customer support documents
with open("data/customer_support_docs.json", "r", encoding="utf-8") as file:
    documents = json.load(file)

index = faiss.read_index("data/faiss_index.idx")
model = SentenceTransformer("all-MiniLM-L6-v2")

# FastAPI app setup
app = FastAPI()

class QueryRequest(BaseModel):
    user_query: str

def retrieve_relevant_info(user_query):
    """Retrieve the most relevant FAQ from FAISS index."""
    query_embedding = model.encode([user_query]).astype("float32")
    _, closest_index = index.search(query_embedding, 1)
    return documents[closest_index[0][0]]["answer"]

@app.get("/")
def read_root():
    return {"message": "AI Customer Support Copilot with FAISS is running!"}

@app.post("/query/")
def get_response(request: QueryRequest):
    try:
        relevant_info = retrieve_relevant_info(request.user_query)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful customer support assistant."},
                {"role": "user", "content": f"User query: {request.user_query}\nRelevant information: {relevant_info}"}
            ]
        )
        return {"response": response["choices"][0]["message"]["content"]}
    except Exception as e:
        return {"error": str(e)}
