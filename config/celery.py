import os
import sys

import dotenv
from celery import Celery

# Prevent macOS fork() SIGSEGV in libpq GSSAPI and Objective-C runtime inside Celery prefork workers
os.environ.setdefault("PGGSSENCMODE", "disable")
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

dotenv.load_dotenv()

# Set the default Django settings module for the 'celery' program.
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

# Create the Celery app with the project name.
app = Celery("config")

# Load any custom config from Django settings, using keys that start with "CELERY_"
app.config_from_object("django.conf:settings", namespace="CELERY")

# On macOS (Darwin), fork() is inherently unsafe with Apple's libxpc / getaddrinfo / SystemConfiguration
# Use threads pool to prevent SIGSEGV crashes during networking / tile downloads
if sys.platform == "darwin":
    app.conf.worker_pool = "threads"
    app.conf.worker_concurrency = 4

# Auto-discover task modules in all installed Django apps.
app.autodiscover_tasks()


@app.task(bind=True)
def debug_task(self):
    print(f"Request: {self.request!r}")
