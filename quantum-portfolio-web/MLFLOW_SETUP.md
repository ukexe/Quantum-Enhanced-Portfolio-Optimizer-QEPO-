# MLflow Setup Guide for QEPO Web Interface

This guide explains how to set up MLflow tracking server to see real runs in the Reporting & Analytics section.

## 🚀 Quick Setup

### 1. Start MLflow Tracking Server

**Option A: Using the provided script (Recommended)**
```bash
cd quantum-portfolio-web/backend
python start_mlflow.py
```

**Option B: Manual MLflow command**
```bash
mlflow ui --host 0.0.0.0 --port 5000
```

### 2. Start the Web Interface Backend

```bash
cd quantum-portfolio-web/backend
python server.py
```

### 3. Start the Frontend

```bash
cd quantum-portfolio-web
npm run dev
```

## 🔧 Detailed Setup

### Prerequisites

1. **Python 3.8+** installed
2. **MLflow** installed: `pip install mlflow>=2.0.0`
3. **Quantum Portfolio Backend** with some runs (optional for testing)

### Step-by-Step Instructions

#### 1. Install MLflow
```bash
pip install mlflow>=2.0.0
```

#### 2. Start MLflow Server
The MLflow server stores experiment data and provides the tracking API:

```bash
# Navigate to the backend directory
cd quantum-portfolio-web/backend

# Start MLflow server
python start_mlflow.py
```

This will:
- Start MLflow UI at `http://localhost:5000`
- Create a local `mlruns` directory for data storage
- Set up the tracking server for the web interface

#### 3. Start Web Interface Backend
In a new terminal:

```bash
cd quantum-portfolio-web/backend
python server.py
```

The backend will:
- Connect to MLflow at `http://localhost:5000`
- Provide API endpoints for the web interface
- Show real MLflow runs in the Reporting section

#### 4. Start Frontend
In another terminal:

```bash
cd quantum-portfolio-web
npm run dev
```

## 🧪 Testing with Sample Data

If you don't have any quantum portfolio runs yet, you can create some test data:

### Option 1: Run Quantum Portfolio Commands
```bash
cd quantum-portfolio
python -m qepo.cli data ingest
python -m qepo.cli optimize
python -m qepo.cli backtest
```

### Option 2: Use MLflow UI to Create Test Runs
1. Go to `http://localhost:5000`
2. Create a new experiment
3. Start a new run
4. Log some metrics and parameters

## 🔍 Verifying the Setup

### 1. Check MLflow Server
- Open `http://localhost:5000` in your browser
- You should see the MLflow UI with experiments and runs

### 2. Check Web Interface
- Open `http://localhost:3000` in your browser
- Go to "Reporting & Analytics" tab
- You should see real runs from MLflow (not mock data)

### 3. Check Backend Logs
The backend server logs will show:
```
INFO: MLflow tracking URI set to: http://localhost:5000
INFO: Retrieved X runs from MLflow
```

## 🐛 Troubleshooting

### Problem: "No runs found" in Reporting section

**Solution:**
1. Ensure MLflow server is running on port 5000
2. Check backend logs for MLflow connection errors
3. Verify MLflow has some runs (check `http://localhost:5000`)

### Problem: Backend fails to start

**Solution:**
1. Install missing dependencies: `pip install -r requirements.txt`
2. Check Python version (3.8+ required)
3. Ensure port 8000 is not in use

### Problem: MLflow connection errors

**Solution:**
1. Verify MLflow server is running: `curl http://localhost:5000`
2. Check MLflow version: `mlflow --version`
3. Restart MLflow server if needed

### Problem: Empty or mock data still showing

**Solution:**
1. Restart the backend server after starting MLflow
2. Check the backend logs for MLflow connection status
3. Verify the MLflow tracking URI is correct

## 🔧 Configuration

### Environment Variables

You can customize the MLflow connection:

```bash
# Set custom MLflow tracking URI
export MLFLOW_TRACKING_URI="http://localhost:5000"

# Start backend with custom URI
python server.py
```

### Custom MLflow Setup

For production or custom setups:

```bash
# Use SQLite backend
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Use PostgreSQL backend
mlflow ui --backend-store-uri postgresql://user:pass@localhost/mlflow

# Use S3 for artifacts
mlflow ui --default-artifact-root s3://my-bucket/mlflow-artifacts
```

## 📊 What You'll See

Once properly set up, the Reporting & Analytics section will show:

- **Real MLflow runs** from your quantum portfolio experiments
- **Actual metrics** like portfolio returns, Sharpe ratios, volatility
- **Real parameters** like risk aversion, cardinality constraints
- **Live data** that updates as you run new experiments
- **Proper run status** (FINISHED, FAILED, RUNNING)

## 🎯 Next Steps

1. **Run some experiments** using the quantum portfolio CLI
2. **View results** in the web interface Reporting section
3. **Generate reports** for specific runs
4. **Compare strategies** using the analytics features

## 📚 Additional Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Quantum Portfolio CLI Guide](../quantum-portfolio/README.md)
- [Web Interface Documentation](./README.md)
