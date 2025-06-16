from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from langchain_core.documents import Document

embeddings = OllamaEmbeddings(model="llama3.2:latest")

client = QdrantClient("localhost:6333")

if not client.collection_exists('demo_collection'):
    client.create_collection(
        collection_name="demo_collection",
        vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
    )

document = Document(
    page_content="Hello what is up!",
    metadata={"source": "https://example.com"}
)

vector_store = QdrantVectorStore(
    client=client,
    collection_name="demo_collection",
    embedding=embeddings,
)

vector_store.add_documents([document])