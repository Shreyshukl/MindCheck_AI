import os
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load env vars
load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# --- Step 1: Setup Flask ---
app = Flask(__name__)

# --- Step 2: Setup LLM ---
def load_llm():
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model="llama-3.1-8b-instant",
        temperature=0.5,
        max_tokens=512
    )

# --- Step 3: Custom Prompt ---
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don’t know the answer, just say you don’t know. Don’t make up an answer.

Context: {context}
Question: {question}

Answer:
"""
def set_custom_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

# --- Step 4: Load FAISS vectorstore ---
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# --- Step 5: Create QA Chain ---
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# --- Flask Routes ---
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_query = request.json.get("query")
    response = qa_chain.invoke({'query': user_query})

    result = response["result"]
    sources = []
    for i, doc in enumerate(response["source_documents"], start=1):
        sources.append({
            "file": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page", "Unknown"),
            "text": doc.page_content[:500]  # show first 500 chars
        })

    return jsonify({"answer": result, "sources": sources})

if __name__ == "__main__":
    app.run(debug=True)
