import os
import streamlit as st
from dotenv import load_dotenv

# LangChain LLM (Gemini)
from langchain_google_genai import ChatGoogleGenerativeAI

# LlamaIndex core components
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondensePlusContextChatEngine

# LlamaIndex integration with ChromaDB
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# LlamaIndex embedding model
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# LlamaIndex wrapper for LangChain LLMs
from llama_index.llms.langchain import LangChainLLM 

def get_collection_metadata(collection):
    """Fetch metadata summary of the Chroma collection."""
    count = collection.count()
    items = collection.get(include=["metadatas", "documents"], limit=5)  # fetch small sample
    ids = items.get("ids", [])
    sample_files = []
    for md in items.get("metadatas", []):
        if md and "file_name" in md:
            sample_files.append(md["file_name"])
    sample_files = list(set(sample_files))[:5]  # unique, top 5

    return {
        "count": count,
        "sample_ids": ids,
        "sample_files": sample_files,
    }


@st.cache_resource
def load_chat_engine():
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise RuntimeError("GOOGLE_API_KEY not set in .env file")

    lc_llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=google_api_key)
    llm = LangChainLLM(lc_llm)
    Settings.llm = llm

    # --- Chroma setup ---
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection("soa_hackathon_rag")
    vector_store = ChromaVectorStore(chroma_collection=collection)

    model_path = os.path.abspath("./bge-m3")
    embed_model = HuggingFaceEmbedding(model_name=model_path)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)

    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=10,
        vector_store_query_mode="hybrid",
        alpha=0.5,
    )

    reranker = SentenceTransformerRerank(
        model="BAAI/bge-reranker-large",
        top_n=3,
    )

    memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

    # --- Dynamic metadata injection ---
    metadata = get_collection_metadata(collection)
    system_prompt = f"""
    You are an expert AWS support assistant.

    Your knowledge comes strictly from:
    1. Retrieved AWS documentation (EC2, S3, Lambda).
    2. Database metadata provided below.

    Database metadata:
    - Total documents stored: {metadata['count']}
    - Example document IDs: {metadata['sample_ids']}
    - Example file names: {metadata['sample_files']}


    Rules:
    - If asked about database size, files, or stored documents, use the metadata.
    - Otherwise, answer strictly from retrieved document context.
    - If insufficient context, say so. Do not invent.
    - Do not provide the unique IDs
    """

    chat_engine = CondensePlusContextChatEngine.from_defaults(
        retriever=retriever,
        node_postprocessors=[reranker],
        memory=memory,
        system_prompt=system_prompt,
    )

    return chat_engine
def main():
    st.set_page_config(page_title="Chatbot Support", layout="wide")
    st.title("Chatbot- powered by Gemini")
    st.markdown("How may I help you today?")

    chat_engine = load_chat_engine()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] =[]
    
    # Display previous messages
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get user input
    if prompt := st.chat_input("Hello, How may I help you today?"):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chat_engine.stream_chat(prompt)
                
                # Stream the response to the UI
                full_response = st.write_stream(response.response_gen)
                
                # The expander for sources will appear after the response is complete
                source_nodes = response.source_nodes
                if source_nodes:
                    with st.expander("See sources"):
                        for i, n in enumerate(source_nodes, 1):
                            meta = n.node.metadata or {}
                            filename = meta.get("file_name", "N/A")
                            page_label = meta.get("page_label", "N/A")
                            st.write(f"**Source {i} from `{filename}` (Page {page_label}):**")
                            st.info(n.node.get_content())
        
        # Add the complete response to the message history
        st.session_state["messages"].append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    main()