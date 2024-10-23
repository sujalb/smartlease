import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import Qdrant as QdrantVectorStore
import chainlit as cl
from langchain_openai import ChatOpenAI
from langchain_qdrant import Qdrant
from langchain.chains import LLMChain
from langchain.schema import Document
import logging
from qdrant_client.http import models as qdrant_models
import re
from langchain.prompts import PromptTemplate


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print(f"Current working directory: {os.getcwd()}")
print(f"Chainlit.md exists: {os.path.exists('chainlit.md')}")

with open('chainlit.md', 'r', encoding='utf-8') as f:
    print(f"Chainlit.md content:\n{f.read()}")


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

# Create a simple prompt template
prompt_template = PromptTemplate(
    input_variables=["query"],
    template="Answer the following question: {query}"
)

# Create an LLMChain with the prompt template
llm_chain = LLMChain(llm=llm, prompt=prompt_template)

# Set favicon and logo paths
#favicon_path = "functions/sl-utils/resources/favicon.ico"
logo_path = "functions/sl-utils/resources/logo.png"

# Ensure the paths exist
# if not os.path.exists(favicon_path):
#     print(f"Warning: Favicon not found at {favicon_path}")
if not os.path.exists(logo_path):
    print(f"Warning: Logo not found at {logo_path}")

# Custom CSS will be applied through chainlit.md

# cl.configure_project(
#     title="SmartLease AI Assistant",
#     description="AI assistant for lease queries",
#     markdown_file="chainlit.md"
# )

@cl.on_message
async def main(message: cl.Message):
    query = message.content
    logger.info(f"Received query: {query}")

    try:
        tenant_name = extract_tenant_name(query)
        logger.info(f"Extracted tenant name: {tenant_name}")
        
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
        
        if not search_result:
            await cl.Message(content=f"No information found for tenant: {tenant_name}").send()
            return

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

        # Include metadata in the prompt, using .get() to provide default values
        metadata_context = f"""Metadata for {tenant_name}:
Lease Start: {metadata_info.get('lease_start', 'Unknown')}
Lease End: {metadata_info.get('lease_end', 'Unknown')}
Rent Amount: {metadata_info.get('rent_amount', 'Unknown')}

"""
        
        structured_prompt = f"""Based on the following context and metadata, provide a response to the question in this format:
1. A clear, concise answer to the question, using the most human-readable date format.
2. List ALL sources of this information, including:
   a. Metadata: Specify the exact metadata field and its value, attributing it to the correct document.
   b. Document content: Provide the document number, page number, and a direct quote if available.
3. If the information is present in multiple sources, report ALL instances.
4. If there are any discrepancies between sources, highlight them.

Always check and report on BOTH the metadata and the document content before responding, even if they contain the same information.

Metadata:
{metadata_context}

Document Context:
{context}

Question: {query}

Structured Answer:"""

        # Use the structured_prompt as the query for the llm_chain
        response = llm_chain.run(query=structured_prompt)

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
