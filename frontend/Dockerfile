FROM node:20-slim

WORKDIR /app

# Step 1: Copy only package.json
COPY frontend/package.json ./

# Step 2: Create lock file *inside* container, on correct architecture
RUN npm install --package-lock-only 

# Step 3: Install deps from this lock file
# Install ALL deps (including vite in devDependencies)
RUN npm install

# Step 4 (optional debug): show vite path
RUN node -p "require.resolve('vite')"
RUN npm list --all
RUN ls -l node_modules/.bin
RUN node -p "require.resolve('vite')"
# Step 5: Copy the rest of your app (but not host node_modules)
COPY frontend/ .

EXPOSE 5173
CMD ["npx", "vite"]