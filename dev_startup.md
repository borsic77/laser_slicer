

⸻

# Laser Slicer – Local Development Setup

This document provides all commands and steps needed to start every required component for developing and running Laser Slicer locally (no Docker).

⸻

Prerequisites
	•	Python 3.10+
	•	Node.js 18+
	•	PostgreSQL (running on localhost:5432)
	•	Redis (running on localhost:6379)

⸻

1. Start PostgreSQL

Start the service:
	•	macOS (Homebrew):
brew services start postgresql
	•	Linux (systemd):
sudo systemctl start postgresql
	•	macOS (Postgres.app):
Open Postgres.app from Applications.

(First time only) Create database and user:

psql postgres

Then in the psql prompt:

CREATE DATABASE laserslicer;
CREATE USER laserslicer WITH PASSWORD ‘localdevpassword’;
GRANT ALL PRIVILEGES ON DATABASE laserslicer TO laserslicer;

⸻

2. Start Redis

Option 1: Foreground
redis-server

Option 2: As a service (macOS Homebrew)
brew services start redis

Verify:
redis-cli ping

Should output: PONG

⸻

3. Start Django Backend

cd ~/laser_slicer
uv venv .venv
source .venv/bin/activate
uv sync
cp .env.example .env         # Make sure your .env is configured for local dev
python manage.py migrate
python manage.py runserver
	•	The backend API is now available at http://localhost:8000

⸻

4. Start Celery Worker

(Open a new terminal, activate your virtual environment again)

cd ~/laser_slicer
source .venv/bin/activate
celery -A config worker -l info

⸻

5. Start Frontend (React + Vite)

(Open another terminal)

cd ~/laser_slicer/frontend
npm install
npm run dev 
	•	The frontend is now available at http://localhost:5173

⸻

Summary Table

Component	Command	Default URL
PostgreSQL	brew services start postgresql	localhost:5432
Redis	redis-server or brew services start redis	localhost:6379
Django backend	python manage.py runserver	http://localhost:8000
Celery worker	celery -A core worker -l info	—
Frontend (Vite)	npm run dev – –host	http://localhost:5173


⸻

All components must be running for the full Laser Slicer stack to work locally.

