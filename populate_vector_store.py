from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from langchain_core.documents import Document

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

client = QdrantClient("localhost:6333")

client.delete_collection("demo_collection")

if not client.collection_exists('demo_collection'):
    client.create_collection(
        collection_name="demo_collection",
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
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