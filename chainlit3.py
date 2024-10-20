import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import Qdrant as QdrantVectorStore
import chainlit as cl
from langchain_openai import ChatOpenAI
from langchain_qdrant import Qdrant
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import Document
import logging
from qdrant_client.http import models as qdrant_models
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Load environment variables
load_dotenv()

# Initialize Qdrant client
openai_api_key = os.getenv("OPENAI_API_KEY")
qdrant_api_key = os.getenv("QDRANT_API_KEY2")
qdrant_url = os.getenv("QDRANT_URL2")

qdrant_client = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_api_key
)

# Initialize OpenAI

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
llm = ChatOpenAI(openai_api_key=openai_api_key)

# Initialize vector store
collection_name = "smartlease_documents"
vector_store = Qdrant(
    client=qdrant_client,
    collection_name=collection_name,
    embeddings=embeddings,
)


# Define a prompt template that encourages using the context
prompt_template = """You are an AI assistant specializing in analyzing lease documents. Use the following context to answer the question. If the answer is found in the context, provide it and cite the source. If not found, say you don't have enough information to answer.

Context:
{context}

Question: {question}

Answer: """

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Create an LLMChain
llm_chain = LLMChain(llm=llm, prompt=PROMPT)

@cl.on_chat_start
async def start():
    await cl.Message(content="Welcome! Ask me anything about AI ethics, regulations, or policies.").send()

@cl.on_message
async def main(message: cl.Message):
    query = message.content
    logger.info(f"Received query: {query}")

    try:
        tenant_name = extract_tenant_name(query)
        
        if not tenant_name:
            await cl.Message(content="Please specify a tenant name in your query.").send()
            return

        search_result = vector_store.similarity_search_with_score(
            query,
            k=5,
            filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="tenant_name",
                        match=qdrant_models.MatchValue(value=tenant_name)
                    )
                ]
            )
        )
        
        docs = []
        s3_urls = set()
        metadata_info = {}
        for doc, score in search_result:
            point_id = doc.metadata['_id']
            qdrant_doc = qdrant_client.retrieve(collection_name, [point_id])[0]
            content = qdrant_doc.payload.get('text', 'No content available')
            metadata = {
                'source': qdrant_doc.payload.get('source', 'Unknown'),
                'page': qdrant_doc.payload.get('page', 'N/A'),
                'tenant_name': qdrant_doc.payload.get('tenant_name', 'Unknown'),
                's3_url': qdrant_doc.payload.get('s3_url', 'No direct link available'),
                'lease_start': qdrant_doc.payload.get('lease_start', 'Unknown'),
                'lease_end': qdrant_doc.payload.get('lease_end', 'Unknown'),
                'rent_amount': qdrant_doc.payload.get('rent_amount', 'Unknown')
            }
            docs.append(Document(page_content=content, metadata=metadata))
            s3_urls.add(metadata['s3_url'])
            metadata_info = metadata  # Store metadata for later use

        context = "\n\n".join([f"Document {i+1} (Source: {doc.metadata['source']}, Page: {doc.metadata['page']}, Tenant: {doc.metadata['tenant_name']}):\n{doc.page_content}" for i, doc in enumerate(docs)])

        # Include metadata in the prompt
        metadata_context = f"Metadata for {tenant_name}:\nLease Start: {metadata_info['lease_start']}\nLease End: {metadata_info['lease_end']}\nRent Amount: {metadata_info['rent_amount']}\n\n"
        
        structured_prompt = f"""Based on the following context and metadata, provide a response to the question in this format:
        1. A clear, concise answer to the question, using both the metadata and document content.
        2. The source of the information (metadata or specific section in the document).
        3. If applicable, the specific section in the document where this information is present.

        Metadata:
        {metadata_context}

        Document Context:
        {context}

        Question: {query}

        Structured Answer:"""

        response = llm_chain.run(context=structured_prompt, question=query)

        # Format the final response
        final_response = f"Query: {query}\n\n{response}\n\nFull lease document(s):"
        for url in s3_urls:
            final_response += f"\n- {url}"

        await cl.Message(content=final_response).send()

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        logger.error(f"Error: {error_message}", exc_info=True)
        await cl.Message(content=error_message).send()

def extract_tenant_name(query):
    # Look for patterns like "tenant John Doe" or "renter Jane Smith"
    pattern = r'\b(tenant|renter|lessee|occupant|resident)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)'
    match = re.search(pattern, query, re.IGNORECASE)
    if match:
        return match.group(2)  # Return the captured name
    return None
