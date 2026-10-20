import react from '@vitejs/plugin-react';
import path from 'path';
import { defineConfig, loadEnv } from 'vite';

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, path.resolve(__dirname, '..'));
  const apiUrl = (env.VITE_API_URL ?? process.env.VITE_API_URL ?? '').replace(/\/+$/, '');

  return {
    plugins: [react()],
    envDir: path.resolve(__dirname, '..'),
    define: {
      'import.meta.env.VITE_API_URL': JSON.stringify(apiUrl),
    },
  };
});
