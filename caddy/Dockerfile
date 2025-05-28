###############################################################################
# Stage 1  – build the React/Vite frontend
FROM node:20-slim AS frontend-builder

WORKDIR /app
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ .
ARG VITE_API_URL
ENV VITE_API_URL=$VITE_API_URL
RUN npm run build               # → /app/dist

###############################################################################
# Stage 2  – final Caddy image that serves /app/dist
FROM caddy:alpine

# copy frontend files into the default doc-root
COPY --from=frontend-builder /app/dist /usr/share/caddy

# copy our site config
COPY Caddyfile /etc/caddy/Caddyfile