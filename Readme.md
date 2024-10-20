# SmartLease File Upload Service

This project sets up an S3 bucket for file uploads using the Serverless Framework.

## Prerequisites

- [Node.js](https://nodejs.org/) (for Serverless Framework)
- [AWS CLI](https://aws.amazon.com/cli/) configured with your credentials
- [Serverless Framework](https://www.serverless.com/) installed globally

## Setup

1. Install Serverless Framework globally:
   ```
   npm install -g serverless
   ```

2. Configure AWS credentials:
   ```
   aws configure
   ```

3. Install project dependencies:
   ```
   npm install
   ```

## Deployment

To deploy the S3 bucket:

```
serverless deploy
```

To deploy to a specific stage:

```
serverless deploy --stage production
```

To deploy to a specific region:

```
serverless deploy --stage production --region us-west-2
```

## List of other commands
```
aws sts get-caller-identity
chainlit run chainlit3.py
```



Questions to ask : 
To get the correct lease information for a specific tenant, you should structure your questions to include the tenant's name. 
Include the word "tenant", "renter", "lessee", "occupant", or "resident" followed by the full name.
The name should have at least two parts (first and last name) and start with capital letters.

Here are some examples of how to phrase your questions:
- "What is the lease start date for tenant James Bennett?"
- "Can you tell me the rent amount for renter Emily Thompson?"
- "When does the lease end for occupant James Bennett?"
- "What are the lease terms for resident Olivia Brooks?"
- "Is there a pet policy for lessee Daniel Brooks?"


Data Storage: 

In the handler.py file, when storing the embeddings, each chunk of text is associated with specific metadata, including the tenant name:

```
   tenant_metadata = base_metadata.copy()
   tenant_metadata.update({
       "tenant_name": tenant,
       "page": text.metadata.get('page', 0),
       "text": text.page_content
   })
```


Querying:
In the chainlit3.py file, when a query is received, the tenant name is extracted and used to filter the search results:

```
   tenant_name = extract_tenant_name(query)
   
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
```

