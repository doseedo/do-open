from celery import Celery
import logging
# Configure Celery with a consistent app name
celery_app = Celery('audio_video_tasks',
                    broker='pyamqp://guest:guest@rabbitmq:5672//',
                    backend='rpc://')

# Ensure tasks are properly imported from both task files
task_acks_late = True
worker_pool = 'threads'
worker_concurrency = 4
celery_app.conf.imports = ('video_tasks', 'audio_tasks')

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define task routes (update as necessary)
celery_app.conf.task_routes = {
    'video_tasks.*': {'queue': 'video_queue'},
    'audio_tasks.*': {'queue': 'audio_queue'}  # Add this if you have specific routing for audio tasks
}



