from fastapi import FastAPI, File, UploadFile
from google.cloud import videointelligence
from google.oauth2 import service_account
import os

app = FastAPI()

@celery_app.task(name='process_video')
def process_video(file_path):
    try:
        # Initialize the Google Cloud Video Intelligence client with credentials
        credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', '/path/to/credentials.json')
        credentials = service_account.Credentials.from_service_account_file(
            credentials_path,
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )
        video_client = videointelligence.VideoIntelligenceServiceClient(credentials=credentials)

        # Perform label detection on the video file
        with open(file_path, 'rb') as video:
            input_content = video.read()
        features = [videointelligence.Feature.LABEL_DETECTION]

        operation = video_client.annotate_video(features=features, input_content=input_content)
        print('\nProcessing video for label annotations:')

        result = operation.result(timeout=90)
        print('\nFinished processing.')

        # Process video analysis results
        segment_labels = result.annotation_results[0].segment_label_annotations
        labels = [segment_label.entity.description for segment_label in segment_labels]

        os.remove(file_path)
        return {"labels": labels}

    except Exception as e:
        return {"error": str(e)}
