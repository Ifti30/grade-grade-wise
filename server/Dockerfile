FROM node:18-bullseye

# Set working directory inside the container
WORKDIR /app

# Install Python runtime and pip (for ML scripts)
RUN apt-get update && apt-get install -y --no-install-recommends \
  python3 python3-pip python3-venv \
  && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (CPU wheels)
RUN pip3 install --no-cache-dir \
  numpy pandas scikit-learn lightgbm torch \
  matplotlib seaborn imbalanced-learn joblib

# Install Node dependencies (need prisma schema available for postinstall)
COPY package*.json ./
COPY prisma ./prisma
RUN npm install

# Copy the rest of the server code
COPY . .

# Ensure startup script is executable
RUN chmod +x /app/scripts/resource-logger.sh

# Expose the API port
EXPOSE 3000

# Start the server in production mode (no file-watching)
ENV NODE_ENV=production
CMD ["sh", "-c", "/app/scripts/resource-logger.sh & npm run start"]
