FROM osgeo/gdal:ubuntu-small-3.6.2

# Install system dependencies (remove build tools after install)
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        python3-pip python3-dev python3-gdal gdal-bin \
        build-essential g++ curl ca-certificates tzdata \
    && pip3 install --upgrade pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only dependency declarations for caching
COPY pyproject.toml poetry.lock* ./

# Install Poetry and dependencies
RUN curl -sSL https://install.python-poetry.org | python3 - \
 && PATH="/root/.local/bin:$PATH" poetry config virtualenvs.create false \
 && PATH="/root/.local/bin:$PATH" poetry install --no-interaction --no-root \
 # Remove build tools to shrink image
 && apt-get remove -y build-essential g++ \
 && apt-get autoremove -y \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.local/bin:$PATH"

# Copy application source
COPY . .

EXPOSE 8000
CMD ["python3", "manage.py", "runserver", "0.0.0.0:8000"]