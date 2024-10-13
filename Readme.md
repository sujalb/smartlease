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
```



Questions to ask : 
what is the address at the lease? 
what is the name of the tenant? 
what is the name of the landlord? 
what is the sq footage of the property? 