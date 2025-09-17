import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load .env so GROQ_API_KEY works
load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# --- Step 1: Setup LLM (Groq) ---
def load_llm():
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model="llama-3.1-8b-instant",
        temperature=0.5,
        max_tokens=512
    )

# --- Step 2: Custom Prompt ---
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don’t know the answer, just say you don’t know. Don’t make up an answer. Don't hallucinate.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
Answer:
"""

def set_custom_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

# --- Step 3: Load FAISS vectorstore ---
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# --- Step 4: Create QA chain ---
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# --- Step 5: Continuous Q&A loop ---
print("Interactive Q&A session started. Type 'exit' to quit.\n")

while True:
    user_query = input("Write Query Here: ")
    if user_query.lower() in ["exit", "quit"]:
        print("Session ended.")
        break

    response = qa_chain.invoke({'query': user_query})

    print("\nRESULT:", response["result"])
    print("\nSOURCE DOCUMENTS:")
    for i, doc in enumerate(response["source_documents"], start=1):
        print(f"\n--- Source {i} ---")
        print("File:", doc.metadata.get("source", "Unknown"))
        print("Page:", doc.metadata.get("page", "Unknown"))
        print("Text Snippet:\n", doc.page_content[:500], "...")  # first 500 chars for readability
    print("\n" + "="*80 + "\n")
