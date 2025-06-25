import re
import csv
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
# embeddings = OllamaEmbeddings(model="rjmalagon/gte-qwen2-1.5b-instruct-embed-f16")
embeddings = OllamaEmbeddings(model="llama3.2:latest")
client = QdrantClient(
    url=qdrant_url, 
    api_key=qdrant_api_key,
    check_compatibility=False,  # Skip compatibility check
    timeout=60  # Increase timeout
)

# Create collection if it doesn't exist
# collection_name = "banking_ai_usecases_small"
collection_name = "banking_ai_usecases"

try:
    # Test connection first
    collections = client.get_collections()
    print(f"Successfully connected to Qdrant. Available collections: {collections}")
    
    if not client.collection_exists(collection_name):
        print(f"Creating collection '{collection_name}'...")
        client.create_collection(
            collection_name=collection_name,
            # vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
        )
    else:
        print(f"Collection '{collection_name}' already exists.")

    # Initialize the vector store
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )

    # Read case studies from TSV file
    documents = []
    with open('FlowFactory_Case Study Database.tsv', 'r', encoding='utf-8') as file:
        # Skip the first line which contains "// filepath:" comment
        next(file)
        
        # Use csv reader with tab delimiter
        reader = csv.reader(file, delimiter='\t')
        
        # Get headers from the first row
        headers = next(reader)
        
        # Process each row in the TSV file
        for i, row in enumerate(reader):
            if not row or all(cell.strip() == '' for cell in row):
                continue  # Skip empty rows
                
            # Extract data from each column
            case_study_title = row[0] if len(row) > 0 else ""
            challenge = row[1] if len(row) > 1 else ""
            solution = row[2] if len(row) > 2 else ""
            overall_impact = row[3] if len(row) > 3 else ""
            key_learnings = row[4] if len(row) > 4 else ""
            future_prospects = row[5] if len(row) > 5 else ""
            
            # Combine all data into a comprehensive document
            content = f"{case_study_title}\n\nChallenge:\n{challenge}\n\n" \
                      f"Solution:\n{solution}\n\n" \
                      f"Overall Impact:\n{overall_impact}\n\n" \
                      f"Key Learnings:\n{key_learnings}\n\n" \
                      f"Future Prospects:\n{future_prospects}"
            
            # Create a title from case study title
            title_match = re.search(r'^Case Study \d+: (.*)', case_study_title)
            title = title_match.group(1).strip() if title_match else f"Banking Use Case {i+1}"
            
            # Create a Document with metadata
            doc = Document(
                page_content=content,
                metadata={
                    "source": "banking_ai_usecases",
                    "title": title,
                    "index": i,
                    "case_study": case_study_title
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
    print("Error: FlowFactory_Case Study Database.tsv file not found. Please make sure the file exists in the current directory.")
except Exception as e:
    print(f"An error occurred: {str(e)}")
    import traceback
    traceback.print_exc()