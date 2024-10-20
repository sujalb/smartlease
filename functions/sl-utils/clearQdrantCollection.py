from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize Qdrant client

qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL2"),
    api_key=os.getenv("QDRANT_API_KEY2")
)

# Define your collection name
COLLECTION_NAME = "smartlease_documents"

try:
    # Delete all points in the collection
    qdrant_client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.FilterSelector(filter=models.Filter())
    )
    print(f"All points in collection '{COLLECTION_NAME}' have been deleted.")

    # Delete processed_docs.json file
    processed_docs_path = os.path.join(os.path.dirname(__file__), 'processed_docs.json')
    if os.path.exists(processed_docs_path):
        os.remove(processed_docs_path)
        print("processed_docs.json has been deleted.")
    else:
        print("processed_docs.json does not exist.")

except Exception as e:
    print(f"An error occurred: {e}")
