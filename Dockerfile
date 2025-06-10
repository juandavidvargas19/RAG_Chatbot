FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the PDF file into the Docker image
COPY random_machine_learing.pdf /app/random_machine_learing.pdf

# Copy application code
COPY Chatbot.py .
COPY Langgraph_Agent.py .
COPY helper_functions.py .
COPY .env* ./

# Create directories for data persistence
RUN mkdir -p /app/emissions_data

# Expose Streamlit port
EXPOSE 8501

# Expose prometheus
EXPOSE 8080

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Accept build argument
ARG ARCHITECTURE_VERSION=3

# Set environment variable from build argument
ENV ARCHITECTURE_VERSION=${ARCHITECTURE_VERSION}

# Run Streamlit app with version parameter
CMD ["sh", "-c", "streamlit run Chatbot.py --server.port=8501 --server.address=0.0.0.0 -- -v ${ARCHITECTURE_VERSION}"]