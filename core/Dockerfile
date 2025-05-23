# core/Dockerfile
FROM python:3.11-slim

# Install system dependencies for Rasterio, GDAL, etc.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gdal-bin libgdal-dev proj-bin && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install uv (Python package manager)
RUN pip install --no-cache-dir uv

# Copy Python project files
COPY pyproject.toml /app/
COPY core /app/core

# Install Python dependencies
RUN uv pip install --system --no-cache .

# Collect static files at build time (if you want)
# RUN python core/manage.py collectstatic --noinput

# Expose Django port
EXPOSE 8000

# Run Django dev server 
CMD ["python", "core/manage.py", "runserver", "0.0.0.0:8000"]