# Use a modern, slim Python image
FROM python:3.11-slim

# Install system dependencies if any are needed
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up a non-root user for security
RUN useradd -m -u 1000 user
WORKDIR /app

# 1. First, install dependencies (for better layer caching)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# 2. Copy the rest of the application
COPY --chown=user . .

# Sets environment variables
ENV PATH="/home/user/.local/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    AUTONOMY_ENV_DB=/app/autonomy_env.db

# Ensure the app starts on the correct port and host
EXPOSE 7860

# Launch the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
