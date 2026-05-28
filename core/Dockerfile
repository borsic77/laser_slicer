FROM osgeo/gdal:ubuntu-full-3.6.2

# Install system dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        python3-pip python3-dev gdal-bin \
        build-essential g++ curl ca-certificates tzdata \
    && pip3 install --upgrade pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv
RUN pip3 install --no-cache-dir uv

# Copy only dependency declarations for layer caching
COPY pyproject.toml uv.lock ./

# Install Python dependencies into the system Python environment.
# UV_SYSTEM_PYTHON=1 mirrors the old `poetry config virtualenvs.create false`
# behaviour so that all executables (celery, gunicorn, …) land on $PATH.
# gdal==3.6.2 is compiled from source; the ubuntu-full base image provides
# the required GDAL 3.6.2 headers and gdal-config.
RUN UV_SYSTEM_PYTHON=1 uv sync --frozen --no-install-project \
    && apt-get remove -y build-essential g++ \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy application source
COPY . .

EXPOSE 8000
CMD ["python3", "manage.py", "runserver", "0.0.0.0:8000"]
