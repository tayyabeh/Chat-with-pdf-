# Use the official Python 3.10 image from Docker Hub
FROM python:3.10-slim  

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy your project files to the container
COPY . /app/

# Set permissions
RUN chmod +x /app/streamlit_app.py

# Expose the port that Streamlit will run on
EXPOSE 8501

# Healthcheck for the container
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit application
CMD ["streamlit", "run", "streamlit_app.py"]