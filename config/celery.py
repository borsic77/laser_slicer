import os

from celery import Celery

# Set the default Django settings module for the 'celery' program.
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

# Create the Celery app with the project name.
app = Celery("config")

# Load any custom config from Django settings, using keys that start with "CELERY_"
app.config_from_object("django.conf:settings", namespace="CELERY")

# Auto-discover task modules in all installed Django apps.
app.autodiscover_tasks()


@app.task(bind=True)
def debug_task(self):
    print(f"Request: {self.request!r}")
