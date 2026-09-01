FROM python:3.10-slim

# Set up a non-root user (Hugging Face Spaces requirement)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Change ownership of /app to the new user
USER root
RUN chown -R user:user /app
USER user

# Install system dependencies needed by OpenCV
USER root
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
USER user

# Copy requirements and install them
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code and models
COPY --chown=user . .

# Expose the default port used by Hugging Face Spaces
EXPOSE 7860

# Start the FastAPI server using uvicorn
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
