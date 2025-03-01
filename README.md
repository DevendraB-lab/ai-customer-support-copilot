<<<<<<< HEAD
# ai-customer-support-copilot
GenAI-powered support copilot using RAG, FAISS, and OpenAI API to improve customer interactions.

## Overview
A Generative AI-powered support copilot using Retrieval-Augmented Generation (RAG), FAISS, and OpenAI API to assist customer support agents in real-time.

## Features
✅ **RAG-based retrieval** from historical support tickets  
✅ **FAISS for vector search** and document similarity  
✅ **LLM-powered query resolution** using OpenAI API  
✅ **FastAPI for serving responses**  
✅ **Real-time chat support**  

## Tech Stack
- **Python** (FastAPI, FAISS, OpenAI API)
- **Vector Database** (FAISS)
- **Large Language Model** (GPT-4 via OpenAI API)
- **MLOps Tools** (MLflow for tracking)
- **Deployment** (Docker, AWS EKS)

## Installation
```powershell
git clone https://github.com/DevendraB-lab/ai-customer-support-copilot.git
cd ai-customer-support-copilot
pip install -r requirements.txt
=======
🚀 AI Customer Support Copilot
Overview
This project is an AI-powered Customer Support Copilot that helps answer customer queries by using Retrieval-Augmented Generation (RAG). It combines FAISS (Facebook AI Similarity Search) for retrieving relevant support documents and GPT-3.5-turbo for generating accurate responses.

✨ Features
✅ RAG-based retrieval using FAISS for relevant support documents.
✅ OpenAI GPT-3.5 integration for intelligent query responses.
✅ FastAPI backend for handling API requests.
✅ FAISS vector search for fast similarity matching.
✅ Embeddings via Sentence-Transformers (all-MiniLM-L6-v2).
✅ Structured FAQ knowledge base for post-sale customer support.
✅ Scalable deployment-ready architecture.

🛠 Tech Stack
Backend: Python, FastAPI
Machine Learning: OpenAI GPT-3.5, FAISS, Sentence-Transformers
Database: FAISS vector database
Deployment: Uvicorn, Docker, AWS (future)
🔨 Step-by-Step Breakdown
1️⃣ Setting Up the Project
📌 Folder Structure
bash
Copy
Edit
ai-customer-support-copilot/
│── data/                     # Stores FAQ data and FAISS index
│   ├── customer_support_docs.json
│   ├── faiss_index.idx
│── src/                      # Source code files
│   ├── app.py                # FastAPI backend
│   ├── faiss_db.py           # FAISS index creation
│── requirements.txt          # Dependencies
│── README.md                 # Documentation
📌 Initial Setup
Clone the Repository
bash
Copy
Edit
git clone https://github.com/your-username/ai-customer-support-copilot.git
cd ai-customer-support-copilot
Create a Virtual Environment (Optional but Recommended)
bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # For Mac/Linux
venv\Scripts\activate     # For Windows
Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
2️⃣ Creating the Knowledge Base
We compiled a structured FAQ dataset related to Post-Sale Customer Support (e.g., warranty, RC transfer, insurance).

📌 File: data/customer_support_docs.json

json
Copy
Edit
[
    {
        "question": "How long does car ownership transfer take?",
        "answer": "Ownership transfer typically takes 30 to 90 days, depending on the RTO."
    },
    {
        "question": "What documents are required for RC transfer?",
        "answer": "You need the original Registration Certificate (RC), insurance, and buyer's ID proof."
    }
]
3️⃣ Building the FAISS Index
We converted the FAQ dataset into vector embeddings using Sentence-Transformers and stored them in a FAISS index for similarity search.

📌 FAISS Index Creation Script (src/faiss_db.py)
python
Copy
Edit
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load data
with open("data/customer_support_docs.json", "r", encoding="utf-8") as file:
    documents = json.load(file)

# Generate embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
question_embeddings = model.encode([doc["question"] for doc in documents])

# Create FAISS index
index = faiss.IndexFlatL2(question_embeddings.shape[1])
index.add(np.array(question_embeddings, dtype="float32"))

# Save index
faiss.write_index(index, "data/faiss_index.idx")
print("FAISS index created successfully!")
Run the script:

bash
Copy
Edit
python src/faiss_db.py
4️⃣ Building the FastAPI Backend
The FastAPI backend handles:

Retrieving relevant FAQ data from FAISS.
Sending the retrieved context to GPT-3.5 for a final response.
📌 API Implementation (src/app.py)
python
Copy
Edit
from fastapi import FastAPI
import openai
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel

# Load API Key
openai.api_key = "your-openai-api-key"

# Load FAISS index & documents
with open("data/customer_support_docs.json", "r") as file:
    documents = json.load(file)
index = faiss.read_index("data/faiss_index.idx")
model = SentenceTransformer("all-MiniLM-L6-v2")

app = FastAPI()

class QueryRequest(BaseModel):
    user_query: str

def retrieve_faq_answer(user_query):
    """Retrieve closest matching FAQ answer from FAISS"""
    query_embedding = model.encode([user_query]).astype("float32")
    _, closest_index = index.search(query_embedding, 1)
    return documents[closest_index[0][0]]["answer"]

@app.get("/")
def root():
    return {"message": "AI Customer Support Copilot is running!"}

@app.post("/query/")
def get_response(request: QueryRequest):
    relevant_info = retrieve_faq_answer(request.user_query)
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful customer support assistant."},
            {"role": "user", "content": f"User query: {request.user_query}\nRelevant info: {relevant_info}"}
        ]
    )
    return {"response": response.choices[0].message.content}
Run the API Server:

bash
Copy
Edit
uvicorn src.app:app --reload
5️⃣ Testing the API
Test using Swagger UI:

Open: http://127.0.0.1:8000/docs
Use the /query/ endpoint and enter:
json
Copy
Edit
{ "user_query": "How do I transfer my car insurance?" }
Expected Response: A relevant answer retrieved from FAISS + GPT-generated text.


📌 Contributors
👨‍💻 Devendra Baghel - AI Product Manager
📧 Contact: bagheldevendra70@gmail.com

📌 Future Improvements
🚀 Add multi-turn conversations using memory storage
🚀 Improve retrieval with a larger knowledge base
🚀 Deploy on AWS Lambda or Render

🚀 Final Words
This AI-powered Customer Support Copilot combines semantic search (FAISS) + OpenAI GPT-3.5 to create an efficient, scalable support assistant.

🔹 Like this project? Give it a ⭐ on GitHub!
>>>>>>> d12c5df (Updated README with code snippets)
