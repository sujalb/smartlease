import os
from typing import List, Any
import chainlit as cl
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain.schema import BaseRetriever
from pydantic import BaseModel, Field
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize OpenAI embedding model
embeddings = OpenAIEmbeddings()

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_HOST"),
    api_key=os.getenv("QDRANT_API_KEY")
)

# Initialize Qdrant as a vector store
vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name="smartlease_documents",
    embedding_function=embeddings.embed_query
)

# Initialize ChatOpenAI model
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

# Custom retriever class
class CustomRetriever(BaseRetriever, BaseModel):
    vectorstore: Any = Field(...)

    def get_relevant_documents(self, query: str) -> List[Document]:
        try:
            # Retrieve documents from Qdrant
            docs = self.vectorstore.similarity_search(query, k=20)  # Retrieve more docs initially
            logger.info(f"Retrieved {len(docs)} documents from Qdrant")
            valid_docs = []
            for i, doc in enumerate(docs):
                logger.info(f"Document {i+1}: type={type(doc)}, page_content={repr(doc.page_content)}, metadata={doc.metadata}")
                # Ensure page_content is valid and non-empty
                if isinstance(doc, Document) and isinstance(doc.page_content, str) and doc.page_content.strip():
                    valid_docs.append(doc)
                    logger.info(f"Valid document {i+1}: {doc.page_content[:50]}...")
                    if len(valid_docs) == 3:  # Limit to 3 valid documents
                        break
                else:
                    logger.warning(f"Invalid document {i+1}: type={type(doc)}, page_content={repr(doc.page_content)}")

            if not valid_docs:
                logger.warning("No valid documents found.")
                return [Document(page_content="No valid information found.")]
            
            logger.info(f"Returning {len(valid_docs)} valid documents")
            return valid_docs
        except Exception as e:
            logger.error(f"Error in get_relevant_documents: {e}", exc_info=True)
            return [Document(page_content="Error retrieving documents.")]

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        return self.get_relevant_documents(query)

# Create an instance of the custom retriever
custom_retriever = CustomRetriever(vectorstore=vector_store)

# Create a retrieval chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=custom_retriever,
    return_source_documents=True,
)

@cl.on_chat_start
async def start():
    await cl.Message("Welcome to the SmartLease Assistant! How can I help you today?").send()

@cl.on_message
async def main(message: cl.Message):
    try:
        # Get the response from the QA chain
        response = qa_chain.invoke({"query": message.content})
        
        # Extract the answer and source documents
        answer = response['result']
        source_docs = response.get('source_documents', [])
        
        # Create the response message
        msg_content = answer + "\n\nSources:\n"
        for i, doc in enumerate(source_docs):
            # Ensure the document has valid content before displaying it
            if isinstance(doc, Document) and isinstance(doc.page_content, str) and doc.page_content.strip():
                msg_content += f"\nSource {i+1}: {doc.page_content[:100]}..."  # Truncate long content
        
        # Send the message to the user
        await cl.Message(content=msg_content).send()

    except Exception as e:
        error_msg = f"An error occurred: {str(e)}"
        logger.error(f"Detailed error: {e}", exc_info=True)
        await cl.Message(content=error_msg).send()

# Log information about the Qdrant collection
logger.info(f"Qdrant collection info: {qdrant_client.get_collection('smartlease_documents')}")
