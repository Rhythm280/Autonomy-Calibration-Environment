# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set up a new user named "user" with UID 1000
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
# We use --chown=user to ensure the new user has permissions
COPY --chown=user . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Make port 7860 available to the world (Hugging Face standard)
EXPOSE 7860

# Define environment variables
ENV AUTONOMY_ENV_DB=/app/autonomy_env.db
ENV PYTHONUNBUFFERED=1

# Run main.py when the container launches
# Hugging Face looks for a process listening on port 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
