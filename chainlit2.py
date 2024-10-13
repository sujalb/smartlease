import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Qdrant
import chainlit as cl
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

# Initialize OpenAI
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(temperature=0)

# Initialize vector store
collection_name = "smartlease_documents"
vector_store = Qdrant(
    client=qdrant_client,
    collection_name=collection_name,
    embeddings=embeddings,
)

# Create a custom prompt template
template = """You are an AI assistant specializing in analyzing lease documents. 
Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Be sure to provide specific details from the lease document when available.

Context: {context}

Question: {question}
Answer:"""

PROMPT = PromptTemplate(
    template=template, input_variables=["context", "question"]
)

# Create a RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

@cl.on_chat_start
async def start():
    await cl.Message(content="Welcome! Ask me anything about the lease document.").send()

@cl.on_message
async def main(message: cl.Message):
    query = message.content
    logger.info(f"Received query: {query}")

    try:
        # Retrieve documents
        docs = vector_store.similarity_search(query, k=5)
        
        # Process documents
        processed_docs = []
        for doc in docs:
            if 'text' in doc.metadata:
                processed_docs.append(Document(page_content=doc.metadata['text'], metadata=doc.metadata))
            elif doc.page_content:
                processed_docs.append(doc)
            else:
                processed_docs.append(Document(page_content="No content available", metadata=doc.metadata))

        # Use the RetrievalQA chain to get the answer
        response = qa_chain.invoke({"query": query, "input_documents": processed_docs})
        answer = response['result']
        source_docs = response['source_documents']

        logger.info(f"Generated response: {answer}")
        
        # Prepare the response message
        response_message = f"{answer}\n\nSources:"
        for i, doc in enumerate(source_docs):
            page = doc.metadata.get('page', 'N/A')
            content = doc.page_content[:500] if doc.page_content else "No content available"
            response_message += f"\n\nSource {i+1} (Page {page}):\n{content}..."

        await cl.Message(content=response_message).send()

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        logger.error(f"Error: {error_message}", exc_info=True)
        await cl.Message(content=error_message).send()

# Log information about the Qdrant collection
collection_info = qdrant_client.get_collection(collection_name)
logger.info(f"Qdrant collection info: {collection_info}")
logger.info(f"Number of points in collection: {collection_info.points_count}")

# Print sample documents
logger.info("Sample documents:")
sample_docs = vector_store.similarity_search("", k=5)
for i, doc in enumerate(sample_docs):
    logger.info(f"Document {i+1}:")
    logger.info(f"Metadata: {doc.metadata}")
    logger.info(f"Content: {doc.page_content[:200]}...")
