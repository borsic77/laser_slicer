FROM osgeo/gdal:ubuntu-small-3.6.2

# Install system dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install --no-install-recommends -y \
    python3-pip python3-dev python3-gdal gdal-bin \
    build-essential g++ \
    curl ca-certificates \
    tzdata \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only dependency declarations for caching
COPY pyproject.toml poetry.lock* ./

RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"
RUN poetry config virtualenvs.create false \
 && poetry install --no-interaction --no-root


# Copy full application source
COPY . .

EXPOSE 8000
CMD ["python3", "manage.py", "runserver", "0.0.0.0:8000"]