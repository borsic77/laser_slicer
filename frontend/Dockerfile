# frontend/Dockerfile
FROM node:20-slim

WORKDIR /app

COPY frontend/package.json frontend/package-lock.json* ./  # or pnpm-lock.yaml/yarn.lock if used

RUN npm install

COPY frontend /app

EXPOSE 5173  # Vite dev server default

# For dev
CMD ["npm", "run", "dev"]

# For production:
# RUN npm run build
# ...then use nginx to serve dist/