# ingest.py
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb

# --- 1. Load the document from the 'data' directory ---
print("Loading documents...")
# This will look for the PDF in the folder named 'data' that we created.
reader = SimpleDirectoryReader(input_dir="./data")
documents = reader.load_data()
print(f"Loaded {len(documents)} document pages.")

# --- 2. Initialize the embedding model ---
print("Initializing embedding model BAAI/bge-m3...")
# This model will convert our text chunks into numerical vectors (embeddings).
# It's a powerful model that understands the meaning behind the text.
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

# --- 3. Create and persist the vector store ---
print("Creating ChromaDB client and collection...")
# This creates a persistent database in a folder named 'chroma_db'.
# This is where our embeddings will be stored.
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("soa_hackathon_rag")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# --- 4. Create the index ---
# This is the main step. LlamaIndex will take our documents,
# use the embedding model to convert them to vectors,
# and store them in our ChromaDB vector store.
print("Creating index... This may take a few minutes on the first run.")
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    embed_model=embed_model,
    show_progress=True
)

print("\nIngestion complete. Vector store created in './chroma_db'")