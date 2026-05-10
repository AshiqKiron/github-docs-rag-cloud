import os
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import pinecone

# 🔑 ENV VARS
PINECONE_API = os.getenv("PINECONE_API_KEY")
GROQ_API = os.getenv("GROQ_API_KEY")
INDEX_NAME = "github-docs-rag"
DOCS_PATH = "./target_docs"

@st.cache_resource
def build_rag_chain():
    pc = pinecone.Pinecone(api_key=PINECONE_API)
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    existing_indexes = pc.list_indexes().names()
    
    if INDEX_NAME not in existing_indexes:
        with st.spinner("📥 Loading docs & building vector index..."):
            pc.create_index(
                name=INDEX_NAME, 
                dimension=384, 
                metric="cosine",
                spec=pinecone.ServerlessSpec(cloud="aws", region="us-east-1")
            )
            
            loader = DirectoryLoader(DOCS_PATH, glob="**/*.md")
            docs = loader.load()
            
            if not docs:
                st.error("❌ No documents found! Check target_docs/ folder.")
                st.stop()
            
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(docs)
            
            PineconeVectorStore.from_documents(chunks, embedder, index_name=INDEX_NAME)
            st.success(f"✅ Indexed {len(chunks)} chunks from {len(docs)} files.")
    else:
        st.info("📂 Vector index loaded.")

    index = pc.Index(INDEX_NAME)
    
    # 🔧 FIX: Increase k to get more results
    retriever = PineconeVectorStore(index=index, embedding=embedder).as_retriever(
        search_kwargs={"k": 5}  # Changed from 3 to 5
    )
    
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3, api_key=GROQ_API)
    
    # 🔧 FIX: More flexible prompt that actually uses the context
    prompt = ChatPromptTemplate.from_template(
        """You are a helpful documentation assistant. Use the provided context to answer the question.

Context:
{context}

Question: {input}

Answer:"""
    )
    
    stuff_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, stuff_chain)

# 🖥️ UI
st.set_page_config(page_title="GitHub Docs RAG", layout="wide")
st.title("📦 GitHub Docs Search")

if not PINECONE_API or not GROQ_API:
    st.error("⚠️ Missing API keys in Settings.")
    st.stop()

rag_chain = build_rag_chain()

# 🔍 Debug: Show how many docs are indexed
with st.expander("🔍 Debug Info"):
    pc = pinecone.Pinecone(api_key=PINECONE_API)
    index = pc.Index(INDEX_NAME)
    stats = index.describe_index_stats()
    st.write(f"**Total vectors in index:** {stats.get('total_vector_count', 0)}")
    st.write(f"**Dimensions:** {stats.get('dimension', 384)}")

query = st.text_input("Ask about the docs:", placeholder="e.g., How do I install?")

if query:
    with st.spinner("🔍 Searching..."):
        result = rag_chain.invoke({"input": query})
        
        # Show answer
        st.subheader("✅ Answer")
        st.write(result["answer"])
        
        # 🔧 FIX: Always show retrieved context
        st.divider()
        st.subheader("📖 Retrieved Context (Top 5)")
        for i, doc in enumerate(result["context"], 1):
            with st.expander(f"Chunk {i} - Source: `{doc.metadata.get('source', 'unknown')}`"):
                st.text(doc.page_content)
