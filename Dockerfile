# =========================================================================
# Dockerfile for APH-IF (Advanced Parallel Hybrid - Intelligent Fusion)
# Project: APH-IF Technology Framework
# Author: Alexander Ricciardi
# Date: 2025-08-04
# =========================================================================

# --- Image Objective ---
# This image provides a production-ready environment for APH-IF applications
# that implement Advanced Parallel Hybrid search capabilities including 
# VectorRAG, GraphRAG, and intelligent context fusion. The container supports
# both Streamlit frontend and FastAPI backend services with Neo4j integration.
# -------------------------------------------------------------------------

# --- Base Image ---
# Using Python 3.12 slim for security, performance, and compatibility
FROM python:3.12-slim

# --- Metadata ---
LABEL maintainer="Alexander Ricciardi <alex.omegapy@gmail.com>"
LABEL description="APH-IF Technology Framework Container"
LABEL version="1.0.0"
LABEL license="Apache-2.0"

# --- Working Directory ---
WORKDIR /app

# --- System Dependencies ---
# Install system packages required for Python packages and health checks
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# --- Python Dependencies ---
# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- Application Setup ---
# Copy application files
COPY . .

# Create necessary directories
RUN mkdir -p /app/data /app/logs

# --- Environment Variables ---
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# --- Ports ---
# Expose ports for Streamlit (8501) and FastAPI (8000)
EXPOSE 8501 8000

# --- Health Check ---
# Basic health check for the container
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# --- Default Command ---
# Start Streamlit by default (can be overridden)
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
