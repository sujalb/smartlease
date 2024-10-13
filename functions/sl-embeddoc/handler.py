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
from qdrant_client.http import models
import uuid

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
if 'AWS_LAMBDA_FUNCTION_NAME' not in os.environ:
    load_dotenv()
    logger.info("Loaded environment variables from .env file")

# Initialize clients
s3 = boto3.client('s3')
logger.info(f"Initialized S3 client. Using region: {s3.meta.region_name}")

qdrant_client = QdrantClient(
    os.environ.get('QDRANT_URL'),
    api_key=os.environ.get('QDRANT_API_KEY')
)
logger.info(f"Initialized Qdrant client. QDRANT_URL: {os.environ.get('QDRANT_URL')}")

# Initialize embeddings model
openai_api_key = os.environ.get('OPENAI_API_KEY')
if not openai_api_key:
    logger.error("OPENAI_API_KEY is not set in the environment variables")
    raise ValueError("OPENAI_API_KEY is not set")

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
logger.info("Initialized OpenAI Embeddings")

def update_processed_file(bucket, key, timestamp):
    processed_file_key = 'post-embed/processed.txt'
    
    try:
        response = s3.get_object(Bucket=bucket, Key=processed_file_key)
        content = response['Body'].read().decode('utf-8')
        logger.info(f"Retrieved existing processed.txt file from S3")
    except s3.exceptions.NoSuchKey:
        content = ""
        logger.info("No existing processed.txt file found in S3")
    
    new_entry = f"{key}, {timestamp}\n"
    updated_content = content + new_entry
    
    s3.put_object(Bucket=bucket, Key=processed_file_key, Body=updated_content)
    logger.info(f"Updated processed.txt file in S3 with new entry: {new_entry.strip()}")

def lambda_handler(event, context):
    logger.info(f"Received event: {json.dumps(event)}")
    
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
    
    # Create base metadata for all chunks
    base_metadata = {
        "source": f"s3://{bucket}/{key}",
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
                models.Record(
                    id=str(uuid.uuid4()),  # Generate a UUID for each record
                    vector=embedding,
                    payload=tenant_metadata
                )
            )

    collection_name = "smartlease_documents"
    logger.info(f"Storing embeddings in Qdrant collection: {collection_name}")
    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
    )

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
    test_event = {
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
            },
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
    
    logger.info(f"Using AWS region: {os.environ.get('AWS_DEFAULT_REGION', 'not set')}")
    
    result = lambda_handler(test_event, None)
    logger.info(f"Lambda handler result: {json.dumps(result, indent=2)}")
