from google.cloud import videointelligence
from celery_config import celery_app
import os

@celery_app.task(name='video_tasks.process_video')
def process_video(file_path):
    try:
        # Initialize Google Cloud Video Intelligence client
        video_client = videointelligence.VideoIntelligenceServiceClient()

        # Perform label detection, shot change detection, and object tracking
        features = [
            videointelligence.Feature.LABEL_DETECTION,
            videointelligence.Feature.SHOT_CHANGE_DETECTION,
            videointelligence.Feature.OBJECT_TRACKING
        ]

        # Read the video file
        with open(file_path, 'rb') as video:
            input_content = video.read()

        # Prepare the request
        request = {
            'features': features,
            'input_content': input_content
        }

        # Start the annotation request
        operation = video_client.annotate_video(request=request)
        print('\nProcessing video for analysis...')

        # Retrieve results
        result = operation.result(timeout=300)  # Increased timeout for complex analysis
        print('\nFinished processing.')

        analysis_data = {
            "labels": [],
            "scene_changes": [],
            "object_tracking": []
        }

        # Process label detection results
        segment_labels = result.annotation_results[0].segment_label_annotations
        analysis_data["labels"] = [label.entity.description for label in segment_labels]

        # Process scene change detection
        shot_annotations = result.annotation_results[0].shot_annotations
        analysis_data["scene_changes"] = [
            shot.start_time_offset.total_seconds() for shot in shot_annotations
        ]

        # Process object tracking results
        object_annotations = result.annotation_results[0].object_annotations
        for annotation in object_annotations:
            object_name = annotation.entity.description
            for track in annotation.tracks:
                for timestamped_object in track.timestamped_objects:
                    time_offset = timestamped_object.time_offset.total_seconds()
                    bbox = timestamped_object.normalized_bounding_box
                    analysis_data["object_tracking"].append({
                        "object": object_name,
                        "timestamp": time_offset,
                        "bounding_box": {
                            "left": bbox.left,
                            "top": bbox.top,
                            "right": bbox.right,
                            "bottom": bbox.bottom
                        }
                    })

        # Return structured JSON for easy integration with other scripts
        return analysis_data

    except Exception as e:
        print(f"Error processing video: {e}")
        return {"error": str(e)}
    finally:
        # Clean up the temporary file
        if os.path.exists(file_path):
            os.remove(file_path)
