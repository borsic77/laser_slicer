import react from '@vitejs/plugin-react';
import path from 'path';
import { defineConfig, loadEnv } from 'vite';

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  // loadEnv reads .env files from the project root (one level above frontend/).
  // When running inside Docker the env var is injected via docker-compose
  // `environment:` rather than a file, so loadEnv won't see it.
  // Fall back to process.env so Docker dev-server works without a .env file.
  const env = loadEnv(mode, path.resolve(__dirname, '..'));
  const apiUrl =
    env.VITE_API_URL ?? process.env.VITE_API_URL ?? 'http://localhost:8000';

  return {
    plugins: [react()],
    define: {
      'import.meta.env.VITE_API_URL': JSON.stringify(apiUrl),
    },
  };
});
