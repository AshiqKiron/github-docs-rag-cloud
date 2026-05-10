import os
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain, create_stuff_documents_chain
import pinecone

# 🔑 ENV VARS (Set in HF Spaces Secrets)
PINECONE_API = os.getenv("PINECONE_API_KEY")
GROQ_API = os.getenv("GROQ_API_KEY")
INDEX_NAME = "github-docs-rag"
DOCS_PATH = "./target_docs"

@st.cache_resource
def build_rag_chain():
    pc = pinecone.Pinecone(api_key=PINECONE_API)
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create index only if missing
    if INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
        pc.create_index(
            name=INDEX_NAME, dimension=384, metric="cosine",
            spec=pinecone.ServerlessSpec(cloud="aws", region="us-east-1")
        )
        with st.spinner("📥 Loading docs & building vector index... (runs once)"):
            loader = DirectoryLoader(DOCS_PATH, glob="**/*.md")
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(docs)
            PineconeVectorStore.from_documents(chunks, embedder, index_name=INDEX_NAME)
            st.success(f"✅ Indexed {len(chunks)} chunks to Pinecone.")
    else:
        st.info("📂 Vector index already built. Loading...")

    index = pc.Index(INDEX_NAME)
    retriever = PineconeVectorStore(index=index, embedding=embedder).as_retriever(search_kwargs={"k": 3})
    
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1, api_key=GROQ_API)
    prompt = ChatPromptTemplate.from_template(
        """Answer strictly using the context. If unknown, say "Not in docs."
        Always cite source files.
        
        Context:
        {context}
        
        Question: {input}"""
    )
    stuff_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, stuff_chain)

# 🖥️ UI
st.set_page_config(page_title="GitHub Docs RAG", layout="wide")
st.title("📦 Cloud GitHub Docs Search")
st.caption("Hosted on Hugging Face • Vector DB: Pinecone • LLM: Groq")

if not PINECONE_API or not GROQ_API:
    st.error("⚠️ Missing API keys. Set `PINECONE_API_KEY` & `GROQ_API_KEY` in HF Space Secrets.")
    st.stop()

rag_chain = build_rag_chain()

query = st.text_input("Ask about the docs:", placeholder="e.g., How to configure middleware?")
if query:
    with st.spinner("🔍 Retrieving & generating..."):
        result = rag_chain.invoke({"input": query})
        st.subheader("✅ Answer")
        st.write(result["answer"])
        
        st.divider()
        st.subheader("📖 Citations & Context")
        for doc in result["context"]:
            st.markdown(f"**File:** `{doc.metadata.get('source', 'unknown')}`")
            st.text(doc.page_content[:250] + "...")
            st.divider()
