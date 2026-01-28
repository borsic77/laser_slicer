# Deployment Guide

## Architecture
Based on your server setup, the application runs as two separate Docker Compose stacks:

1. **Backend Stack** (`/srv/docker/laser_slicer/`)
   - Contains: Django (backend), Celery, Redis, Postgres
   - **Responsibility**: API, Database, Background Tasks
   - **File**: `docker-compose.prod.yml`

2. **Frontend & Proxy Stack** (`/srv/docker/reverse-proxy/`)
   - Contains: Caddy (Frontend + Reverse Proxy)
   - **Responsibility**: Builds React app, SSL, Routing
   - **File**: `docker-compose.yml` (in `reverse-proxy` dir)

## How to Deploy Updates

### 1. Update Code
```bash
cd /srv/docker/laser_slicer
git pull
```

### 2. Update Backend (if Python/Django changes)
```bash
cd /srv/docker/laser_slicer
docker compose -f docker-compose.prod.yml up -d --build
```
> **Note**: Use `--no-cache` if you suspect old files are sticking around.

### 3. Update Frontend (if React/TS changes)
Since the frontend is built inside the `caddy` image in the *reverse-proxy* stack, you must rebuild that container to see UI changes.

```bash
cd /srv/docker/reverse-proxy
# Force rebuild of the caddy image which contains the frontend build
docker compose build --no-cache caddy
docker compose up -d caddy
```
