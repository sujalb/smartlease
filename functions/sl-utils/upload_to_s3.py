import boto3
import os
from botocore.exceptions import ClientError

def upload_file(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        print(f"Error uploading file {file_name}: {e}")
        return False
    return True

def upload_pdfs_from_folder(folder_path, bucket):
    """Upload all PDF files from a folder to an S3 bucket

    :param folder_path: Path to the folder containing PDF files
    :param bucket: S3 bucket to upload to
    """
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(folder_path, filename)
            if upload_file(file_path, bucket):
                print(f"Successfully uploaded {filename} to {bucket}")
            else:
                print(f"Failed to upload {filename}")

if __name__ == "__main__":
    # Replace with your local folder path containing PDF files
    local_folder = "./resources/"
    
    # S3 bucket name
    bucket_name = "smartlease-uploads-dev"

    upload_pdfs_from_folder(local_folder, bucket_name)