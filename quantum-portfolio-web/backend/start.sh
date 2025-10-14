#!/bin/bash

# Start the QEPO Web API server
echo "Starting QEPO Web API Server..."

# Install dependencies if needed
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Start the server
echo "Starting server on http://localhost:8000"
python server.py
