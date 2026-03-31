FROM python:3.10-slim

WORKDIR /app

# Copy the entire project for multi-mode context
COPY . .

# Install dependencies from the server subfolder
RUN pip install --no-cache-dir -r server/requirements.txt
RUN pip install --no-cache-dir .

# Expose the OpenEnv port
EXPOSE 7860

# Start the server using the absolute module path
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
