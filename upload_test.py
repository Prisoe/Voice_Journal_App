from datetime import datetime
import time
import requests

import boto3
import os
import uuid
from dotenv import load_dotenv


load_dotenv()


# AWS Configuration
S3_BUCKET = os.getenv('TRANSCRIBE_BUCKET_NAME')
AWS_REGION = os.getenv('AWS_REGION')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
DYNAMODB_TABLE = os.getenv('DYNAMODB_TABLE_NAME')



# Initialize AWS clients
s3_client = boto3.client(
    's3',
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

transcribe_client = boto3.client(
    'transcribe',
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

dynamodb = boto3.resource(
    'dynamodb',
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)
table = dynamodb.Table(DYNAMODB_TABLE)

def upload_file_to_s3(file_path):
    """Uploads an audio file to S3 and returns the S3 URI."""
    file_name = f"{uuid.uuid4()}-{os.path.basename(file_path)}"
    try:
        s3_client.upload_file(file_path, S3_BUCKET, file_name)
        s3_uri = f"s3://{S3_BUCKET}/{file_name}"
        print(f"File uploaded successfully to S3: {s3_uri}")
        return s3_uri
    except Exception as e:
        print(f"Error uploading file: {e}")
        return None


def save_to_dynamodb(audio_uri, transcription):
    """Saves the audio URI and transcription to DynamoDB"""
    entry_id = str(uuid.uuid4())
    try:
        table.put_item(
            Item={
                "entry_id": entry_id,
                "timestamp": datetime.utcnow().isoformat(),
                "audio_uri": audio_uri,
                "transcription": transcription
            }
        )
        print(f"Successfully saved to DynamoDB with entry_id: {entry_id}")
    except Exception as e:
        print(f'Error saving to DynamoDB: {e}')


def transcribe_audio(file_uri):
    """Starts a transcription job and returns the transcription file URI."""
    job_name = f"transcribe-job-{uuid.uuid4()}"
    try:
        # Start the transcription job
        transcribe_client.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={"MediaFileUri": file_uri},
            MediaFormat=file_uri.split('.')[-1],  # Infer format from URI
            LanguageCode="en-US",
        )
        print(f"Started transcription job: {job_name}")

        # Poll for the job status
        while True:
            status = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
            job_status = status['TranscriptionJob']['TranscriptionJobStatus']
            if job_status in ["COMPLETED", "FAILED"]:
                break
            print("Transcription job in progress...")
            time.sleep(5)  # Wait before polling again

        if job_status == "COMPLETED":
            transcription_uri = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
            print(f"Transcription completed. File URI: {transcription_uri}")


            # Save to Dynamo DB
            response = requests.get(transcription_uri)
            response.raise_for_status()
            transcription_data = response.json()
            transcription_text = transcription_data["results"]["transcripts"][0]["transcript"]

            save_to_dynamodb(file_uri, transcription_text)


            return transcription_uri

        else:
            print('Transcription job failed.')
            return None
    except Exception as e:
        print(f"An error occured during transcription: {e}")
        return None

if __name__ == "__main__":
    # Input the file path
    file_path = input("Enter the path of the audio file to upload: ").strip()

    if os.path.exists(file_path):
        # Upload the file and get the S3 URI
        s3_uri = upload_file_to_s3(file_path)
        if s3_uri:
            # Trigger transcription and get the transcription URI
            transcription_uri = transcribe_audio(s3_uri)
            if transcription_uri:
                print(f"Transcription available at: {transcription_uri}")
            else:
                print("Failed to transcribe the audio file.")
        else:
            print("Failed to upload the file to S3.")
    else:
        print("File not found. Please check the path and try again.")