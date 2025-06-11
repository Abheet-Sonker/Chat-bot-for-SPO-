import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

# Load environment variables for local development
load_dotenv()

# Load API key from Streamlit secrets or .env
groq_api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("GROQ_API_KEY not found in environment variables or Streamlit secrets!")
    st.stop()

# Load embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

# Load FAISS index
@st.cache_resource
def load_vectorstore():
    return FAISS.load_local(
        "faiss_index",
        embedding_model,
        allow_dangerous_deserialization=True
    )

vectorstore = load_vectorstore()

# Initialize LLM (Groq)
llm = ChatGroq(
    temperature=0.1,
    model_name="llama3-8b-8192",
    groq_api_key=groq_api_key
)

# Build QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "lambda_mul": 0.4}
    ),
    return_source_documents=True
)

# Streamlit UI
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
