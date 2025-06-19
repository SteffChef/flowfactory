import re
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from dotenv import load_dotenv
import os
load_dotenv()
env = os.environ

from langchain_core.documents import Document

# Get API key from environment
qdrant_api_key = env.get("QDRANT_API_KEY")
if not qdrant_api_key:
    raise ValueError("QDRANT_API_KEY not found in environment variables")
qdrant_url = env.get("QDRANT_URL")
if not qdrant_url:
    raise ValueError("QDRANT_URL not found in environment variables")

# Initialize embeddings and client
# embeddings = OllamaEmbeddings(model="llama3.2:latest")
embeddings = OllamaEmbeddings(model="rjmalagon/gte-qwen2-1.5b-instruct-embed-f16")
client = QdrantClient(
    url=qdrant_url, 
    api_key=qdrant_api_key,
    check_compatibility=False,  # Skip compatibility check
    timeout=60  # Increase timeout
)

# Create collection if it doesn't exist
collection_name = "banking_ai_usecases_small"

try:
    # Test connection first
    collections = client.get_collections()
    print(f"Successfully connected to Qdrant. Available collections: {collections}")
    
    if not client.collection_exists(collection_name):
        print(f"Creating collection '{collection_name}'...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
    else:
        print(f"Collection '{collection_name}' already exists.")

    # Initialize the vector store
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )

    # Read case studies from file
    with open('case_studies.txt', 'r') as file:
        text = file.read()
    
    # Use regex to split the text into case studies
    case_studies = re.split(r'\n\s*\n(?=Case Study \d+:)', text)
    
    # Process each case study into a Document
    documents = []
    for i, study in enumerate(case_studies):
        if study.strip():
            # Remove "Case Study N:" prefix
            cleaned_study = re.sub(r'^Case Study \d+:\s*', '', study.strip())
            
            # Extract the bank name and use case title for metadata
            title_match = re.search(r'^([^:]+):', cleaned_study)
            title = title_match.group(1).strip() if title_match else f"Banking Use Case {i+1}"
            
            # Create a Document with metadata
            doc = Document(
                page_content=cleaned_study,
                metadata={
                    "source": "banking_ai_usecases",
                    "title": title,
                    "index": i
                }
            )
            documents.append(doc)
    
    # Add documents to vector store
    if documents:
        print(f"Adding {len(documents)} case studies to Qdrant...")
        vector_store.add_documents(documents)
        print("Successfully added documents to vector store.")
    else:
        print("No case studies found in the file.")

except FileNotFoundError:
    print("Error: case_studies.txt file not found. Please make sure the file exists in the current directory.")
except Exception as e:
    print(f"An error occurred: {str(e)}")
    import traceback
    traceback.print_exc()