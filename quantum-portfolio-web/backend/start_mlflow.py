#!/usr/bin/env python3
"""
Start MLflow tracking server for the Quantum Portfolio Optimizer.
This script starts the MLflow UI server that the web interface connects to.
"""

import subprocess
import sys
import os
from pathlib import Path

def start_mlflow_server():
    """Start MLflow tracking server."""
    # Default MLflow tracking URI
    mlflow_uri = "http://localhost:5000"
    
    # Check if MLflow is installed
    try:
        import mlflow
        print(f"✓ MLflow is installed (version: {mlflow.__version__})")
    except ImportError:
        print("❌ MLflow is not installed. Please install it with:")
        print("   pip install mlflow>=2.0.0")
        sys.exit(1)
    
    # Set up MLflow tracking directory
    mlflow_dir = Path("mlruns")
    mlflow_dir.mkdir(exist_ok=True)
    
    print(f"🚀 Starting MLflow tracking server...")
    print(f"   Tracking URI: {mlflow_uri}")
    print(f"   Tracking Directory: {mlflow_dir.absolute()}")
    print(f"   Web UI: {mlflow_uri}")
    print()
    print("Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Start MLflow server
        subprocess.run([
            sys.executable, "-m", "mlflow", "ui",
            "--host", "0.0.0.0",
            "--port", "5000",
            "--backend-store-uri", f"file://{mlflow_dir.absolute()}",
            "--default-artifact-root", f"{mlflow_dir.absolute()}/artifacts"
        ], check=True)
    except KeyboardInterrupt:
        print("\n🛑 MLflow server stopped")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start MLflow server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_mlflow_server()
