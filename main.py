import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import boto3
from botocore.exceptions import ClientError

app = FastAPI()

# Configure AWS credentials and S3 bucket
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')
S3_BUCKET = os.getenv('S3_BUCKET')
S3_REGION = os.getenv('S3_REGION', 'us-east-1')

s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=S3_REGION
)

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    if file.content_type not in ["application/pdf", "application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
        return JSONResponse(status_code=400, content={"message": "Only PDF and DOC files are allowed"})
    
    try:
        s3_client.upload_fileobj(file.file, S3_BUCKET, file.filename)
        return JSONResponse(status_code=200, content={"message": f"File {file.filename} uploaded successfully"})
    except ClientError as e:
        return JSONResponse(status_code=500, content={"message": f"Error uploading file: {str(e)}"})

@app.get("/")
async def root():
    return {"message": "Welcome to the File Upload API"}