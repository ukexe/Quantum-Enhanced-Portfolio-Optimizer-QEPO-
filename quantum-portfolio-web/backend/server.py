#!/usr/bin/env python3
"""
Simple FastAPI backend server for the Quantum Portfolio Optimizer Web Interface.
This server provides REST API endpoints and WebSocket support for real-time updates.
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="QEPO Web API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for demo purposes
jobs_db: Dict[str, Dict] = {}
configs_db: Dict[str, Dict] = {}
mlflow_runs_db: List[Dict] = []

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # Remove disconnected clients
                self.active_connections.remove(connection)

manager = ConnectionManager()

# Pydantic models
class DataIngestionRequest(BaseModel):
    history: Dict[str, str]
    universe: Dict[str, Any]
    cache: Dict[str, Any]

class OptimizationRequest(BaseModel):
    solver: Dict[str, Any]
    objective: Dict[str, Any]
    constraints: Dict[str, Any]
    hardware: Dict[str, Any]

class BacktestRequest(BaseModel):
    strategy: str
    rebalance: str
    window: Dict[str, int]
    costs_bps: float
    benchmark: str
    rolling: bool
    constraints: Dict[str, Any]
    objective: Dict[str, Any]

class ReportRequest(BaseModel):
    run_id: str
    format: str
    include_charts: bool

class ConfigRequest(BaseModel):
    name: str
    content: Dict[str, Any]

# Utility functions
def generate_pdf_charts_content(charts_data: Dict[str, Any]) -> str:
    """Generate chart content optimized for PDF format."""
    return f"""
    <h2>Performance Charts</h2>
    <div class="charts-container">
        <div class="chart-section">
            <h3>Equity Curve</h3>
            <table class="chart-table">
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Portfolio Value</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join([f'<tr><td>{d["date"]}</td><td>{d["value"]:.3f}</td></tr>' for d in charts_data.get('equity_curve', [])])}
                </tbody>
            </table>
        </div>
        <div class="chart-section">
            <h3>Monthly Returns</h3>
            <table class="chart-table">
                <thead>
                    <tr>
                        <th>Month</th>
                        <th>Return (%)</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join([f'<tr><td>{d["month"]}</td><td>{d["return"]*100:.2f}</td></tr>' for d in charts_data.get('monthly_returns', [])])}
                </tbody>
            </table>
        </div>
        <div class="chart-section">
            <h3>Asset Allocation</h3>
            <table class="chart-table">
                <thead>
                    <tr>
                        <th>Asset</th>
                        <th>Weight (%)</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join([f'<tr><td>{d["asset"]}</td><td>{d["weight"]*100:.2f}</td></tr>' for d in charts_data.get('asset_allocation', [])])}
                </tbody>
            </table>
        </div>
        <div class="chart-section">
            <h3>Risk Metrics</h3>
            <table class="chart-table">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join([f'<tr><td>{d["metric"]}</td><td>{d["value"]:.3f}</td></tr>' for d in charts_data.get('risk_metrics', [])])}
                </tbody>
            </table>
        </div>
    </div>"""

def generate_charts_content(charts_data: Dict[str, Any], format_type: str) -> str:
    """Generate chart content based on format."""
    if format_type == "html":
        return f"""
    <h2>Performance Charts</h2>
    <div class="charts-container">
        <div class="chart-section">
            <h3>Equity Curve</h3>
            <canvas id="equityCurve" width="400" height="200"></canvas>
        </div>
        <div class="chart-section">
            <h3>Monthly Returns</h3>
            <canvas id="monthlyReturns" width="400" height="200"></canvas>
        </div>
        <div class="chart-section">
            <h3>Asset Allocation</h3>
            <canvas id="assetAllocation" width="400" height="200"></canvas>
        </div>
        <div class="chart-section">
            <h3>Risk Metrics</h3>
            <canvas id="riskMetrics" width="400" height="200"></canvas>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
        // Equity Curve Chart
        const equityData = {charts_data.get('equity_curve', [])};
        new Chart(document.getElementById('equityCurve'), {{
            type: 'line',
            data: {{
                labels: equityData.map(d => d.date),
                datasets: [{{
                    label: 'Portfolio Value',
                    data: equityData.map(d => d.value),
                    borderColor: '#3498db',
                    backgroundColor: 'rgba(52, 152, 219, 0.1)',
                    tension: 0.1
                }}]
            }},
            options: {{
                responsive: true,
                scales: {{
                    y: {{
                        beginAtZero: false,
                        title: {{
                            display: true,
                            text: 'Portfolio Value'
                        }}
                    }}
                }}
            }}
        }});
        
        // Monthly Returns Chart
        const monthlyData = {charts_data.get('monthly_returns', [])};
        new Chart(document.getElementById('monthlyReturns'), {{
            type: 'bar',
            data: {{
                labels: monthlyData.map(d => d.month),
                datasets: [{{
                    label: 'Monthly Return (%)',
                    data: monthlyData.map(d => d.return * 100),
                    backgroundColor: monthlyData.map(d => d.return >= 0 ? '#27ae60' : '#e74c3c'),
                    borderColor: monthlyData.map(d => d.return >= 0 ? '#27ae60' : '#e74c3c'),
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                scales: {{
                    y: {{
                        title: {{
                            display: true,
                            text: 'Return (%)'
                        }}
                    }}
                }}
            }}
        }});
        
        // Asset Allocation Chart
        const allocationData = {charts_data.get('asset_allocation', [])};
        new Chart(document.getElementById('assetAllocation'), {{
            type: 'doughnut',
            data: {{
                labels: allocationData.map(d => d.asset),
                datasets: [{{
                    data: allocationData.map(d => d.weight * 100),
                    backgroundColor: [
                        '#3498db', '#e74c3c', '#f39c12', '#27ae60', '#9b59b6',
                        '#1abc9c', '#34495e', '#e67e22', '#95a5a6', '#f1c40f'
                    ]
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    legend: {{
                        position: 'bottom'
                    }}
                }}
            }}
        }});
        
        // Risk Metrics Chart
        const riskData = {charts_data.get('risk_metrics', [])};
        new Chart(document.getElementById('riskMetrics'), {{
            type: 'radar',
            data: {{
                labels: riskData.map(d => d.metric),
                datasets: [{{
                    label: 'Risk Metrics',
                    data: riskData.map(d => d.value),
                    borderColor: '#e74c3c',
                    backgroundColor: 'rgba(231, 76, 60, 0.2)',
                    pointBackgroundColor: '#e74c3c'
                }}]
            }},
            options: {{
                responsive: true,
                scales: {{
                    r: {{
                        beginAtZero: true
                    }}
                }}
            }}
        }});
    </script>"""
    elif format_type == "markdown":
        return f"""
## Performance Charts

### Equity Curve
```
Date       | Portfolio Value
-----------|----------------
{chr(10).join([f"{d['date']} | {d['value']:.3f}" for d in charts_data.get('equity_curve', [])])}
```

### Monthly Returns
```
Month | Return (%)
------|-----------
{chr(10).join([f"{d['month']}   | {d['return']*100:6.2f}" for d in charts_data.get('monthly_returns', [])])}
```

### Asset Allocation
```
Asset  | Weight (%)
-------|-----------
{chr(10).join([f"{d['asset']:<6} | {d['weight']*100:6.2f}" for d in charts_data.get('asset_allocation', [])])}
```

### Risk Metrics
```
Metric        | Value
--------------|-------
{chr(10).join([f"{d['metric']:<13} | {d['value']:.3f}" for d in charts_data.get('risk_metrics', [])])}
```
"""
    else:  # PDF or other formats
        return f"""
PERFORMANCE CHARTS

Equity Curve:
{chr(10).join([f"{d['date']}: {d['value']:.3f}" for d in charts_data.get('equity_curve', [])])}

Monthly Returns:
{chr(10).join([f"{d['month']}: {d['return']*100:.2f}%" for d in charts_data.get('monthly_returns', [])])}

Asset Allocation:
{chr(10).join([f"{d['asset']}: {d['weight']*100:.2f}%" for d in charts_data.get('asset_allocation', [])])}

Risk Metrics:
{chr(10).join([f"{d['metric']}: {d['value']:.3f}" for d in charts_data.get('risk_metrics', [])])}
"""

def create_job(job_type: str, config: Dict[str, Any]) -> str:
    """Create a new job and return its ID."""
    job_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    job = {
        "id": job_id,
        "type": job_type,
        "status": "pending",
        "startTime": datetime.now().isoformat(),
        "progress": 0,
        "message": f"Starting {job_type} job...",
        "config": config,
        "results": None,
        "logs": [f"[{timestamp}] Job created and queued for execution"]
    }
    jobs_db[job_id] = job
    return job_id

async def simulate_job_progress(job_id: str, job_type: str):
    """Simulate job progress for demo purposes."""
    job = jobs_db.get(job_id)
    if not job:
        return

    # Add log entry when job starts running
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    jobs_db[job_id]["logs"].append(f"[{timestamp}] Job started running")

    # Simulate different job types
    if job_type == "data":
        steps = [
            "Fetching S&P 500 universe from Wikipedia...",
            "Downloading historical prices from yfinance...",
            "Computing returns and covariance matrices...",
            "Saving data to cache...",
            "Data ingestion completed successfully!"
        ]
    elif job_type == "optimize":
        steps = [
            "Loading historical data...",
            "Building QUBO problem...",
            "Initializing QAOA solver...",
            "Running quantum optimization...",
            "Post-processing solution...",
            "Optimization completed successfully!"
        ]
    elif job_type == "backtest":
        steps = [
            "Loading historical data...",
            "Setting up walk-forward analysis...",
            "Running backtest iterations...",
            "Computing performance metrics...",
            "Generating results...",
            "Backtest completed successfully!"
        ]
    else:
        steps = ["Processing...", "Completed!"]

    for i, step in enumerate(steps):
        if job_id not in jobs_db:
            break
            
        progress = int((i + 1) / len(steps) * 100)
        
        # Add log entry with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {step}"
        jobs_db[job_id]["logs"].append(log_entry)
        
        jobs_db[job_id].update({
            "progress": progress,
            "message": step,
            "status": "running" if progress < 100 else "completed"
        })
        
        if progress == 100:
            jobs_db[job_id]["endTime"] = datetime.now().isoformat()
            # Add realistic results based on job configuration
            if job_type == "optimize":
                config = jobs_db[job_id]["config"]
                solver_type = config.get("solver", {}).get("type", "qaoa")
                
                # Generate realistic results based on solver type
                if solver_type == "qaoa":
                    portfolio_return = 0.142  # Quantum advantage
                    portfolio_volatility = 0.165
                    sharpe_ratio = 0.86
                elif solver_type == "mvo":
                    portfolio_return = 0.128
                    portfolio_volatility = 0.172
                    sharpe_ratio = 0.74
                else:  # greedy
                    portfolio_return = 0.115
                    portfolio_volatility = 0.185
                    sharpe_ratio = 0.62
                
                cardinality_k = config.get("constraints", {}).get("cardinality_k", 25)
                max_weight = config.get("constraints", {}).get("weight_bounds", [0, 0.1])[1]
                
                # Generate top holdings based on realistic stock data
                top_holdings = [
                    {"ticker": "AAPL", "weight": min(max_weight, 0.085)},
                    {"ticker": "MSFT", "weight": min(max_weight, 0.078)},
                    {"ticker": "GOOGL", "weight": min(max_weight, 0.072)},
                    {"ticker": "AMZN", "weight": min(max_weight, 0.068)},
                    {"ticker": "TSLA", "weight": min(max_weight, 0.065)},
                    {"ticker": "META", "weight": min(max_weight, 0.062)},
                    {"ticker": "NVDA", "weight": min(max_weight, 0.058)},
                    {"ticker": "BRK.B", "weight": min(max_weight, 0.055)},
                    {"ticker": "UNH", "weight": min(max_weight, 0.052)},
                    {"ticker": "JNJ", "weight": min(max_weight, 0.048)}
                ]
                
                jobs_db[job_id]["results"] = {
                    "portfolio_return": portfolio_return,
                    "portfolio_volatility": portfolio_volatility,
                    "sharpe_ratio": sharpe_ratio,
                    "num_selected_assets": cardinality_k,
                    "max_weight": max_weight,
                    "top_holdings": top_holdings[:min(10, cardinality_k)]
                }
            elif job_type == "backtest":
                config = jobs_db[job_id]["config"]
                strategy = config.get("strategy", "qaoa")
                rebalance = config.get("rebalance", "monthly")
                
                # Generate realistic backtest results based on strategy
                if strategy == "qaoa":
                    total_return = 0.487
                    annualized_return = 0.142
                    volatility = 0.165
                    sharpe_ratio = 0.86
                    max_drawdown = 0.072
                    avg_turnover = 0.18
                elif strategy == "mvo":
                    total_return = 0.432
                    annualized_return = 0.128
                    volatility = 0.172
                    sharpe_ratio = 0.74
                    max_drawdown = 0.085
                    avg_turnover = 0.22
                else:  # greedy
                    total_return = 0.398
                    annualized_return = 0.115
                    volatility = 0.185
                    sharpe_ratio = 0.62
                    max_drawdown = 0.092
                    avg_turnover = 0.15
                
                # Benchmark performance (SPY-like)
                benchmark_return = 0.356
                benchmark_volatility = 0.182
                benchmark_sharpe = 0.68
                
                # Generate equity curve data
                equity_curve = []
                monthly_returns = []
                drawdowns = []
                
                import random
                random.seed(42)  # For consistent results
                
                # Generate 12 months of data
                portfolio_value = 1.0
                benchmark_value = 1.0
                peak_value = 1.0
                
                for month in range(12):
                    # Portfolio monthly return
                    portfolio_monthly = (annualized_return / 12) + random.gauss(0, volatility / (12**0.5))
                    portfolio_value *= (1 + portfolio_monthly)
                    
                    # Benchmark monthly return
                    benchmark_monthly = (benchmark_return / 12) + random.gauss(0, benchmark_volatility / (12**0.5))
                    benchmark_value *= (1 + benchmark_monthly)
                    
                    # Update peak and calculate drawdown
                    peak_value = max(peak_value, portfolio_value)
                    drawdown = (peak_value - portfolio_value) / peak_value
                    
                    equity_curve.append({
                        "date": f"2024-{month+1:02d}-01",
                        "portfolio": portfolio_value,
                        "benchmark": benchmark_value
                    })
                    
                    monthly_returns.append({
                        "month": f"2024-{month+1:02d}",
                        "portfolio": portfolio_monthly,
                        "benchmark": benchmark_monthly
                    })
                    
                    drawdowns.append({
                        "date": f"2024-{month+1:02d}-01",
                        "drawdown": drawdown
                    })
                
                jobs_db[job_id]["results"] = {
                    "total_return": total_return,
                    "annualized_return": annualized_return,
                    "volatility": volatility,
                    "sharpe_ratio": sharpe_ratio,
                    "max_drawdown": max_drawdown,
                    "avg_turnover": avg_turnover,
                    "benchmark_return": benchmark_return,
                    "benchmark_volatility": benchmark_volatility,
                    "benchmark_sharpe": benchmark_sharpe,
                    "equity_curve": equity_curve,
                    "monthly_returns": monthly_returns,
                    "drawdowns": drawdowns
                }
        
        # Broadcast update
        await manager.broadcast(json.dumps({
            "type": "job_update",
            "job_id": job_id,
            "job": jobs_db[job_id]
        }))
        
        await asyncio.sleep(2)  # Simulate processing time

# API Routes

@app.get("/")
async def root():
    return {"message": "QEPO Web API Server", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Data Ingestion API
@app.post("/data")
async def start_data_ingestion(request: DataIngestionRequest):
    """Start data ingestion job."""
    job_id = create_job("data", request.dict())
    
    # Start background task
    asyncio.create_task(simulate_job_progress(job_id, "data"))
    
    return {
        "job_id": job_id,
        "status": "started",
        "message": "Data ingestion job started"
    }

@app.get("/data/status/{job_id}")
async def get_data_status(job_id: str):
    """Get data ingestion job status."""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return jobs_db[job_id]

# Portfolio Optimization API
@app.post("/optimize")
async def start_optimization(request: OptimizationRequest):
    """Start portfolio optimization job."""
    job_id = create_job("optimize", request.dict())
    
    # Start background task
    asyncio.create_task(simulate_job_progress(job_id, "optimize"))
    
    return {
        "job_id": job_id,
        "status": "started",
        "message": "Portfolio optimization job started"
    }

@app.get("/optimize/status/{job_id}")
async def get_optimization_status(job_id: str):
    """Get optimization job status."""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return jobs_db[job_id]

# Backtesting API
@app.post("/backtest")
async def start_backtest(request: BacktestRequest):
    """Start backtesting job."""
    job_id = create_job("backtest", request.dict())
    
    # Start background task
    asyncio.create_task(simulate_job_progress(job_id, "backtest"))
    
    return {
        "job_id": job_id,
        "status": "started",
        "message": "Backtesting job started"
    }

@app.get("/backtest/status/{job_id}")
async def get_backtest_status(job_id: str):
    """Get backtest job status."""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return jobs_db[job_id]

# Reporting API
@app.post("/report/generate")
async def generate_report(request: ReportRequest):
    """Generate report for a specific run."""
    # Get report data including charts
    report_data = await get_report_data(request.run_id)
    
    # Generate charts content if requested
    charts_content = ""
    if request.include_charts:
        charts_content = generate_charts_content(report_data["charts"], request.format)
    
    # Mock report generation
    if request.format == "html":
        report_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QEPO Report - {request.run_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .summary {{ background: #f8f9fa; padding: 20px; border-radius: 5px; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
        .metric {{ background: white; padding: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .holdings {{ background: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .holding {{ display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #eee; }}
        .charts-container {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; margin: 20px 0; }}
        .chart-section {{ background: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; }}
        .chart-section h3 {{ margin-top: 0; color: #2c3e50; }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; }}
    </style>
</head>
<body>
    <h1>QEPO Report for Run {request.run_id}</h1>
    
    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Run ID:</strong> {request.run_id}</p>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Format:</strong> {request.format}</p>
        <p><strong>Include Charts:</strong> {request.include_charts}</p>
    </div>

    <h2>Performance Metrics</h2>
    <div class="metrics">
        <div class="metric">
            <h3>Total Return</h3>
            <p style="font-size: 24px; color: #27ae60; font-weight: bold;">12.5%</p>
        </div>
        <div class="metric">
            <h3>Volatility</h3>
            <p style="font-size: 24px; color: #e74c3c; font-weight: bold;">18.2%</p>
        </div>
        <div class="metric">
            <h3>Sharpe Ratio</h3>
            <p style="font-size: 24px; color: #3498db; font-weight: bold;">0.69</p>
        </div>
        <div class="metric">
            <h3>Max Drawdown</h3>
            <p style="font-size: 24px; color: #f39c12; font-weight: bold;">8.5%</p>
        </div>
    </div>

    <h2>Top Holdings</h2>
    <div class="holdings">
        <div class="holding">
            <span>AAPL</span>
            <span>8.0%</span>
        </div>
        <div class="holding">
            <span>MSFT</span>
            <span>7.5%</span>
        </div>
        <div class="holding">
            <span>GOOGL</span>
            <span>6.8%</span>
        </div>
        <div class="holding">
            <span>AMZN</span>
            <span>5.2%</span>
        </div>
        <div class="holding">
            <span>TSLA</span>
            <span>4.1%</span>
        </div>
    </div>
{charts_content}
    <div class="footer">
        <p>This is a mock report for demonstration purposes.</p>
        <p>Generated by QEPO (Quantum Enhanced Portfolio Optimizer)</p>
    </div>
</body>
</html>"""
    elif request.format == "markdown":
        report_content = f"""# QEPO Report for Run {request.run_id}

## Summary
- **Run ID**: {request.run_id}
- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Format**: {request.format}
- **Include Charts**: {request.include_charts}

## Performance Metrics
- **Total Return**: 12.5%
- **Volatility**: 18.2%
- **Sharpe Ratio**: 0.69
- **Max Drawdown**: 8.5%

## Top Holdings
1. AAPL: 8.0%
2. MSFT: 7.5%
3. GOOGL: 6.8%
4. AMZN: 5.2%
5. TSLA: 4.1%
{charts_content}
---
*This is a mock report for demonstration purposes.*
*Generated by QEPO (Quantum Enhanced Portfolio Optimizer)*"""
    else:  # PDF format - generate HTML optimized for PDF
        report_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QEPO Report - {request.run_id}</title>
    <style>
        @page {{
            margin: 0.5in;
            size: A4;
        }}
        body {{ 
            font-family: Arial, sans-serif; 
            margin: 0; 
            padding: 20px; 
            line-height: 1.4; 
            color: #333;
            font-size: 12px;
        }}
        h1 {{ 
            color: #2c3e50; 
            border-bottom: 2px solid #3498db; 
            margin-bottom: 20px;
            font-size: 24px;
        }}
        h2 {{ 
            color: #34495e; 
            margin-top: 25px; 
            margin-bottom: 15px;
            font-size: 18px;
        }}
        h3 {{
            color: #2c3e50;
            margin-top: 15px;
            margin-bottom: 10px;
            font-size: 14px;
        }}
        .summary {{ 
            background: #f8f9fa; 
            padding: 15px; 
            border-radius: 5px; 
            margin-bottom: 20px;
            border: 1px solid #dee2e6;
        }}
        .metrics {{ 
            display: grid; 
            grid-template-columns: repeat(2, 1fr); 
            gap: 10px; 
            margin: 15px 0; 
        }}
        .metric {{ 
            background: white; 
            padding: 12px; 
            border-radius: 5px; 
            box-shadow: 0 1px 3px rgba(0,0,0,0.1); 
            border: 1px solid #e9ecef;
        }}
        .holdings {{ 
            background: white; 
            padding: 15px; 
            border-radius: 5px; 
            box-shadow: 0 1px 3px rgba(0,0,0,0.1); 
            border: 1px solid #e9ecef;
        }}
        .holding {{ 
            display: flex; 
            justify-content: space-between; 
            padding: 6px 0; 
            border-bottom: 1px solid #eee; 
        }}
        .holding:last-child {{
            border-bottom: none;
        }}
        .charts-container {{ 
            display: grid; 
            grid-template-columns: repeat(2, 1fr); 
            gap: 15px; 
            margin: 20px 0; 
        }}
        .chart-section {{ 
            background: white; 
            padding: 15px; 
            border-radius: 5px; 
            box-shadow: 0 1px 3px rgba(0,0,0,0.1); 
            text-align: center; 
            border: 1px solid #e9ecef;
            page-break-inside: avoid;
        }}
        .chart-section h3 {{ 
            margin-top: 0; 
            color: #2c3e50; 
            font-size: 14px;
        }}
        .chart-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        .chart-table th, .chart-table td {{
            border: 1px solid #ddd;
            padding: 6px;
            text-align: left;
            font-size: 10px;
        }}
        .chart-table th {{
            background-color: #f8f9fa;
            font-weight: bold;
        }}
        .footer {{ 
            margin-top: 30px; 
            padding-top: 15px; 
            border-top: 1px solid #ddd; 
            color: #666; 
            font-size: 10px;
        }}
        .page-break {{
            page-break-before: always;
        }}
        @media print {{
            body {{ margin: 0; }}
            .page-break {{ page-break-before: always; }}
        }}
    </style>
</head>
<body>
    <h1>QEPO Report for Run {request.run_id}</h1>
    
    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Run ID:</strong> {request.run_id}</p>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Format:</strong> {request.format}</p>
        <p><strong>Include Charts:</strong> {request.include_charts}</p>
    </div>

    <h2>Performance Metrics</h2>
    <div class="metrics">
        <div class="metric">
            <h3>Total Return</h3>
            <p style="font-size: 18px; color: #27ae60; font-weight: bold;">12.5%</p>
        </div>
        <div class="metric">
            <h3>Volatility</h3>
            <p style="font-size: 18px; color: #e74c3c; font-weight: bold;">18.2%</p>
        </div>
        <div class="metric">
            <h3>Sharpe Ratio</h3>
            <p style="font-size: 18px; color: #3498db; font-weight: bold;">0.69</p>
        </div>
        <div class="metric">
            <h3>Max Drawdown</h3>
            <p style="font-size: 18px; color: #f39c12; font-weight: bold;">8.5%</p>
        </div>
    </div>

    <h2>Top Holdings</h2>
    <div class="holdings">
        <div class="holding">
            <span>AAPL</span>
            <span>8.0%</span>
        </div>
        <div class="holding">
            <span>MSFT</span>
            <span>7.5%</span>
        </div>
        <div class="holding">
            <span>GOOGL</span>
            <span>6.8%</span>
        </div>
        <div class="holding">
            <span>AMZN</span>
            <span>5.2%</span>
        </div>
        <div class="holding">
            <span>TSLA</span>
            <span>4.1%</span>
        </div>
    </div>
{generate_pdf_charts_content(report_data["charts"]) if request.include_charts else ""}
    <div class="footer">
        <p>This is a mock report for demonstration purposes.</p>
        <p>Generated by QEPO (Quantum Enhanced Portfolio Optimizer)</p>
    </div>
</body>
</html>"""
    
    # Return as proper file response
    from fastapi.responses import Response
    
    # Set appropriate content type and filename
    if request.format == "html":
        media_type = "text/html"
        filename = f"qepo-report-{request.run_id}.html"
        disposition = f"attachment; filename={filename}"
    elif request.format == "markdown":
        media_type = "text/markdown"
        filename = f"qepo-report-{request.run_id}.md"
        disposition = f"attachment; filename={filename}"
    elif request.format == "pdf":
        media_type = "text/html"  # PDF is HTML optimized for browser PDF conversion
        filename = f"qepo-report-{request.run_id}.html"  # Use .html extension for PDF
        disposition = f"inline; filename={filename}"  # Use inline instead of attachment
    else:
        media_type = "text/plain"
        filename = f"qepo-report-{request.run_id}.txt"
        disposition = f"attachment; filename={filename}"
    
    return Response(
        content=report_content,
        media_type=media_type,
        headers={"Content-Disposition": disposition}
    )

@app.get("/report/{run_id}")
async def get_report_data(run_id: str):
    """Get report data for a specific run."""
    # Mock report data
    return {
        "run_id": run_id,
        "title": f"QEPO Report - {run_id}",
        "summary": {
            "total_return": 0.125,
            "annualized_return": 0.12,
            "volatility": 0.182,
            "sharpe_ratio": 0.69,
            "max_drawdown": 0.085,
            "num_assets": 25
        },
        "charts": {
            "equity_curve": [
                {"date": "2023-01-01", "value": 1.0},
                {"date": "2023-02-01", "value": 1.05},
                {"date": "2023-03-01", "value": 1.08},
                {"date": "2023-04-01", "value": 1.12},
                {"date": "2023-05-01", "value": 1.15}
            ],
            "monthly_returns": [
                {"month": "Jan", "return": 0.05},
                {"month": "Feb", "return": 0.03},
                {"month": "Mar", "return": 0.04},
                {"month": "Apr", "return": 0.03},
                {"month": "May", "return": 0.02}
            ],
            "asset_allocation": [
                {"asset": "AAPL", "weight": 0.08},
                {"asset": "MSFT", "weight": 0.075},
                {"asset": "GOOGL", "weight": 0.068},
                {"asset": "AMZN", "weight": 0.052},
                {"asset": "TSLA", "weight": 0.041}
            ],
            "risk_metrics": [
                {"metric": "Sharpe Ratio", "value": 0.69},
                {"metric": "Sortino Ratio", "value": 0.85},
                {"metric": "Max Drawdown", "value": 0.085},
                {"metric": "VaR (95%)", "value": 0.025}
            ]
        },
        "details": {
            "strategy": "QAOA",
            "parameters": {
                "risk_aversion": 0.5,
                "cardinality_k": 25,
                "qaoa_depth": 3
            },
            "constraints": {
                "no_short": True,
                "weight_bounds": [0.0, 0.1]
            },
            "performance_attribution": {
                "stock_selection": 0.08,
                "sector_allocation": 0.03,
                "market_timing": 0.01
            }
        }
    }

# MLflow API
@app.get("/mlflow/runs")
async def get_mlflow_runs():
    """Get MLflow runs."""
    # Mock MLflow runs
    runs = [
        {
            "run_id": "run-001",
            "start_time": "2024-01-15T10:00:00Z",
            "end_time": "2024-01-15T10:05:00Z",
            "status": "FINISHED",
            "command": "optimize",
            "metrics": {
                "portfolio_return": 0.125,
                "portfolio_volatility": 0.182,
                "sharpe_ratio": 0.69
            },
            "params": {
                "risk_aversion": "0.5",
                "cardinality_k": "25"
            },
            "tags": {
                "mlflow.runName": "QAOA Optimization",
                "strategy": "qaoa"
            },
            "artifacts": ["portfolio.csv", "config.yml"]
        },
        {
            "run_id": "run-002",
            "start_time": "2024-01-15T11:00:00Z",
            "end_time": "2024-01-15T11:10:00Z",
            "status": "FINISHED",
            "command": "backtest",
            "metrics": {
                "total_return": 0.45,
                "annualized_return": 0.12,
                "sharpe_ratio": 0.75
            },
            "params": {
                "strategy": "mvo",
                "rebalance": "monthly"
            },
            "tags": {
                "mlflow.runName": "MVO Backtest",
                "strategy": "mvo"
            },
            "artifacts": ["backtest.csv", "equity_curve.png"]
        }
    ]
    
    return runs

@app.get("/mlflow/runs/{run_id}")
async def get_mlflow_run(run_id: str):
    """Get specific MLflow run."""
    runs = await get_mlflow_runs()
    for run in runs:
        if run["run_id"] == run_id:
            return run
    
    raise HTTPException(status_code=404, detail="Run not found")

# Configuration API
@app.get("/configs")
async def get_configs():
    """Get all configurations."""
    return [
        {
            "name": "data.yml",
            "content": {
                "universe": {"source": ["wikipedia", "kaggle"]},
                "history": {"start": "2018-01-01", "end": "2025-01-01"},
                "cache": {"enable": True, "path": "data/interim"}
            },
            "lastModified": "2024-01-15T09:00:00Z",
            "isValid": True,
            "errors": []
        },
        {
            "name": "optimizer.yml",
            "content": {
                "objective": {"risk_aversion": 0.5},
                "constraints": {"cardinality_k": 25},
                "solver": {"type": "qaoa", "depth": 3}
            },
            "lastModified": "2024-01-15T09:30:00Z",
            "isValid": True,
            "errors": []
        }
    ]

@app.get("/configs/{config_name}")
async def get_config(config_name: str):
    """Get specific configuration."""
    configs = await get_configs()
    for config in configs:
        if config["name"] == config_name:
            return config["content"]
    
    raise HTTPException(status_code=404, detail="Configuration not found")

@app.post("/configs")
async def save_config(request: ConfigRequest):
    """Save configuration."""
    configs_db[request.name] = request.content
    return {"message": "Configuration saved successfully"}

@app.post("/configs/validate")
async def validate_config(config: Dict[str, Any]):
    """Validate configuration."""
    # Mock validation
    errors = []
    if "objective" in config and "risk_aversion" in config["objective"]:
        if config["objective"]["risk_aversion"] < 0 or config["objective"]["risk_aversion"] > 10:
            errors.append("risk_aversion must be between 0 and 10")
    
    return {
        "isValid": len(errors) == 0,
        "errors": errors
    }

# Job Management API
@app.get("/jobs")
async def get_jobs():
    """Get all jobs."""
    return list(jobs_db.values())

@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """Get specific job."""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return jobs_db[job_id]

@app.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancel a job."""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    
    jobs_db[job_id]["status"] = "cancelled"
    jobs_db[job_id]["endTime"] = datetime.now().isoformat()
    
    # Broadcast update
    await manager.broadcast(json.dumps({
        "type": "job_update",
        "job_id": job_id,
        "job": jobs_db[job_id]
    }))
    
    return {"message": "Job cancelled successfully"}

@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job."""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    
    del jobs_db[job_id]
    return {"message": "Job deleted successfully"}

@app.get("/jobs/{job_id}/logs")
async def get_job_logs(job_id: str):
    """Get job logs."""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return jobs_db[job_id]["logs"]

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
