# Quantum-Enhanced Portfolio Optimizer (QEPO)

A fully open-source, zero-cost quantum portfolio optimizer that runs end-to-end in Python using Qiskit simulators, IBM Quantum free tier, yfinance data, and MLflow tracking.

## 🚀 Quickstart

### Installation

```bash
# Clone and set up environment
git clone <repository-url>
cd quantum-portfolio
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up development environment (optional)
make setup-dev
```

### Basic Usage

```bash
# Start MLflow UI
mlflow ui --port 5000

# Download market data
python -m qepo.cli data ingest

# Run quantum portfolio optimization
python -m qepo.cli optimize

# Backtest the strategy
python -m qepo.cli backtest

# Generate comprehensive report
python -m qepo.cli report --run-id <RUN_ID>
```

## 🛠️ Development

### Available Commands

```bash
make help          # Show all available commands
make install-dev   # Install development dependencies
make test          # Run test suite
make test-cov      # Run tests with coverage
make lint          # Run linting checks
make format        # Format code with black/isort
make dev           # Run full dev checks (format + lint + test)
make clean         # Clean build artifacts
```

### Code Quality

The project uses:
- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking
- **pytest** for testing
- **pre-commit** hooks for automated checks

### Test Coverage

Current test coverage: **74%** with 150+ tests passing
- Unit tests for all core modules
- Integration tests for end-to-end workflows
- Quantum simulator tests
- Mock tests for external dependencies

## 📊 Features

### Core Functionality
- **Data Ingestion**: S&P 500 scraping + yfinance integration
- **Quantum Optimization**: QAOA solver with Qiskit Aer simulators
- **Classical Baselines**: MVO and greedy heuristics for comparison
- **Backtesting**: Walk-forward framework with transaction costs
- **Reporting**: Comprehensive reports with charts and metrics
- **IBM Quantum**: Real hardware integration with safety limits

### Zero-Cost Stack
- **Qiskit Aer**: High-performance quantum simulators
- **IBM Quantum Free Tier**: Access to real quantum hardware
- **yfinance**: Free market data
- **MLflow**: Open-source MLOps platform
- **Google Colab**: Free cloud computing (optional)

## 🏗️ Architecture

See `/.cursor/context.md` for detailed architecture, data flow, and module responsibilities.

## 📈 Performance

- **Test Suite**: 150+ tests, 74% coverage
- **Quantum Solvers**: QAOA with configurable depth and shots
- **Classical Baselines**: Mean-variance optimization + greedy heuristics
- **Backtesting**: Walk-forward with configurable rebalancing
- **Hardware**: IBM Quantum integration with 10-minute time limits

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run `make dev` to ensure code quality
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.
