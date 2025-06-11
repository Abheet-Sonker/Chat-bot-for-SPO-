import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

groq_api_key=st.secrets["GROQ_API_KEY"]
# === Load API Key from .env ===
load_dotenv()

# === Load Embedding Model ===
embedding_model = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2", device="cpu")


# === Load FAISS Index ===
@st.cache_resource
def load_vectorstore():
    return FAISS.load_local(
        "faiss_index",
        embedding_model,
        allow_dangerous_deserialization=True
    )

vectorstore = load_vectorstore()

# === Initialize LLM (Groq) ===
llm = ChatGroq(
    temperature=0.1,
    model_name="llama3-8b-8192",  # Groq model name
    groq_api_key="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"  # Your Groq API key
    )

# === Build QA Chain ===
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "lambda_mul": 0.4}
    ),
    return_source_documents=True
)

# === Streamlit UI ===
st.set_page_config(page_title="SPO Chatbot", layout="wide")
st.title("🤖 SPO Chatbot for Placement Proformas")
st.write("Ask any question based on company proformas.")

query = st.text_input("💬 Ask a question:")

if query:
    with st.spinner("Thinking..."):
        response = qa_chain.invoke(query)

        # Show answer
        st.markdown("### ✅ Answer")
        st.success(response["result"])

        # Show sources
        st.markdown("### 📚 Source Documents")
        for doc in response["source_documents"]:
            source = doc.metadata.get("source", "Unknown")
            section = doc.metadata.get("section", "Unknown")
            st.markdown(f"**Source**: `{source}` | **Section**: `{section}`")
            st.code(doc.page_content[:500])
