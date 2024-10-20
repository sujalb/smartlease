import boto3
import os
import json
import logging
from datetime import datetime
from langchain.document_loaders import S3FileLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from dotenv import load_dotenv
import tempfile
from qdrant_client.http import models as rest
import uuid
import urllib.parse
from qdrant_client.http.exceptions import UnexpectedResponse
import requests

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Get Qdrant URL and API key from environment variables
qdrant_url = os.getenv('QDRANT_URL2')
qdrant_api_key = os.getenv('QDRANT_API_KEY2')

# Log the values to verify
logger.info(f"QDRANT_URL from .env: {qdrant_url}")
logger.info(f"QDRANT_API_KEY: {'Set' if qdrant_api_key else 'Not Set'}")

# Initialize clients
s3 = boto3.client('s3')
logger.info(f"Initialized S3 client. Using region: {s3.meta.region_name}")

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_api_key
)
logger.info(f"Initialized Qdrant client with URL: {qdrant_url}")

# Initialize embeddings model
openai_api_key = os.environ.get('OPENAI_API_KEY')
if not openai_api_key:
    logger.error("OPENAI_API_KEY is not set in the environment variables")
    raise ValueError("OPENAI_API_KEY is not set")

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
logger.info("Initialized OpenAI Embeddings")

def update_processed_file(bucket, key, timestamp):
    """
    Update or create a 'processed.txt' file in S3 to keep track of processed files.

    This function maintains a record of files that have been processed by the embedding
    system. It either updates an existing 'processed.txt' file or creates a new one if
    it doesn't exist. Each entry in the file consists of the processed file's key and
    the timestamp of processing.

    Args:
        bucket (str): The name of the S3 bucket where the processed.txt file is stored.
        key (str): The key (path) of the file that was just processed.
        timestamp (str): The timestamp when the file was processed.

    Returns:
        None

    Raises:
        botocore.exceptions.ClientError: If there's an issue with S3 operations.
    """
    processed_file_key = 'post-embed/processed.txt'
    
    try:
        # Attempt to retrieve the existing processed.txt file from S3
        response = s3.get_object(Bucket=bucket, Key=processed_file_key)
        content = response['Body'].read().decode('utf-8')
        logger.info(f"Retrieved existing processed.txt file from S3")
    except s3.exceptions.NoSuchKey:
        # If the file doesn't exist, start with an empty content
        content = ""
        logger.info("No existing processed.txt file found in S3")
    
    # Create a new entry for the processed file
    new_entry = f"{key}, {timestamp}\n"
    
    # Append the new entry to the existing content
    updated_content = content + new_entry
    
    # Upload the updated content back to S3
    s3.put_object(Bucket=bucket, Key=processed_file_key, Body=updated_content)
    logger.info(f"Updated processed.txt file in S3 with new entry: {new_entry.strip()}")

def test_qdrant_connection():
    try:
        # This should return a list of collection names
        collections = qdrant_client.get_collections()
        logger.info(f"Successfully connected to Qdrant. Available collections: {collections}")
        return True
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {str(e)}")
        return False

def create_collection_if_not_exists(url, api_key, collection_name):
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }
    
    # Check if collection exists
    check_url = f"{url}/collections/{collection_name}"
    logger.info(f"Checking if collection exists: {check_url}")
    response = requests.get(check_url, headers=headers)
    
    if response.status_code == 404:
        logger.info(f"Collection {collection_name} does not exist. Creating new collection...")
        create_url = f"{url}/collections/{collection_name}"
        create_payload = {
            "vectors": {
                "size": 1536,
                "distance": "Cosine"
            }
        }
        create_response = requests.put(create_url, json=create_payload, headers=headers)
        if create_response.status_code == 200:
            logger.info(f"Successfully created collection {collection_name}")
        else:
            logger.error(f"Failed to create collection. Status: {create_response.status_code}, Response: {create_response.text}")
            raise Exception(f"Failed to create collection: {create_response.text}")
    elif response.status_code == 200:
        logger.info(f"Collection {collection_name} already exists.")
    else:
        logger.error(f"Unexpected response when checking collection. Status: {response.status_code}, Response: {response.text}")
        raise Exception(f"Unexpected response when checking collection: {response.text}")

def lambda_handler(event, context):
    # Checking qdrant connection
    if not test_qdrant_connection():
        raise Exception("Unable to connect to Qdrant. Please check your configuration.")

    logger.info(f"####Received event ####: {json.dumps(event)}")
    
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']
    metadata = event['Records'][0]['s3']['object'].get('metadata', {})
    logger.info(f"Processing file: {key} from bucket: {bucket}")
    logger.info(f"File metadata: {metadata}")
    
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        s3.download_fileobj(bucket, key, temp_file)
        temp_file_path = temp_file.name
    logger.info(f"Downloaded file to temporary path: {temp_file_path}")

    loader = PyPDFLoader(temp_file_path)
    documents = loader.load()
    logger.info(f"Loaded {len(documents)} pages from PDF")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    logger.info(f"Split documents into {len(texts)} text chunks")
    
    logger.info("Creating embeddings...")
    embeddings_list = embeddings.embed_documents([text.page_content for text in texts])
    logger.info(f"Created {len(embeddings_list)} embeddings")
    
    # Generate the S3 URL
    s3_url = f"https://{bucket}.s3.amazonaws.com/{urllib.parse.quote(key)}"

    # Create base metadata for all chunks
    base_metadata = {
        "source": f"s3://{bucket}/{key}",
        "s3_url": s3_url,  # Add the S3 URL here
        "lease_start": metadata.get('lease_start', ''),
        "lease_end": metadata.get('lease_end', ''),
        "rent_amount": metadata.get('rent_amount', '')
    }

    # Create records for each tenant and each text chunk
    records = []
    for tenant in metadata.get('tenant_names', []):
        for i, (embedding, text) in enumerate(zip(embeddings_list, texts)):
            tenant_metadata = base_metadata.copy()
            tenant_metadata.update({
                "tenant_name": tenant,
                "page": text.metadata.get('page', 0),
                "text": text.page_content
            })
            records.append(
                rest.Record(
                    id=str(uuid.uuid4()),  # Generate a UUID for each record
                    vector=embedding,
                    payload=tenant_metadata
                )
            )

    collection_name = "smartlease_documents"
    logger.info(f"Storing embeddings in Qdrant collection: {collection_name}")

    create_collection_if_not_exists(qdrant_url, qdrant_api_key, collection_name)

    qdrant_client.upload_records(
        collection_name=collection_name,
        records=records,
    )
    logger.info(f"Stored {len(records)} records in Qdrant")
    
    timestamp = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
    update_processed_file(bucket, key, timestamp)
    
    logger.info(f"Processing completed for file: {key}")
    return {
        'statusCode': 200,
        'body': json.dumps(f'Processed and stored embeddings for {key}')
    }

if __name__ == "__main__":
    test_event1 = {
        "Records": [
            {
                "s3": {
                    "bucket": {"name": "smartlease-uploads-dev"},
                    "object": {
                        "key": "resi1.pdf",
                        "metadata": {
                            "tenant_names": ["Emily Thompson", "James Bennett"],
                            "lease_start": "2024-05-01",
                            "lease_end": "2024-04-30",
                            "rent_amount": "$2422.00"
                        }
                    }
                }
            }
        ]
    }
    result = lambda_handler(test_event1, None)

    test_event2 = {
        "Records": [            
            {
                "s3": {
                    "bucket": {"name": "smartlease-uploads-dev"},
                    "object": {
                        "key": "resi2.pdf",
                        "metadata": {
                            "tenant_names": ["Olivia Brooks", "Daniel Brooks"],
                            "lease_start": "2024-07-31",
                            "lease_end": "2026-04-30",
                            "rent_amount": "$2575.00"
                        }
                    }
                }
            }
        ]
    }

    result = lambda_handler(test_event2, None)
    
    logger.info(f"Using AWS region: {os.environ.get('AWS_DEFAULT_REGION', 'not set')}")
    
    
    logger.info(f"Lambda handler result: {json.dumps(result, indent=2)}")
