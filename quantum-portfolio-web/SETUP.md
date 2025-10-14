# Quantum Portfolio Optimizer - Web Interface Setup Guide

This guide will help you set up and run the Quantum Portfolio Optimizer web interface.

## 🚀 Quick Start

### Prerequisites
- **Node.js 18+** - [Download here](https://nodejs.org/)
- **Python 3.11+** - [Download here](https://python.org/)
- **Git** - [Download here](https://git-scm.com/)

### 1. Clone and Setup Frontend

```bash
# Navigate to the web interface directory
cd quantum-portfolio-web

# Install dependencies
npm install

# Create environment file
cp .env.example .env.local
```

### 2. Setup Backend API Server

```bash
# Navigate to backend directory
cd backend

# On Windows
start.bat

# On Linux/Mac
chmod +x start.sh
./start.sh
```

The backend server will start on `http://localhost:8000`

### 3. Start the Frontend

```bash
# In the quantum-portfolio-web directory
npm run dev
```

The web interface will be available at `http://localhost:3000`

### 4. Setup QEPO Backend (Optional)

For full functionality, you'll need the actual QEPO backend:

```bash
# Navigate to the quantum-portfolio directory
cd ../quantum-portfolio

# Install QEPO dependencies
pip install -r requirements.txt

# Run data ingestion
python -m qepo.cli data ingest

# Run optimization
python -m qepo.cli optimize
```

### 5. Setup MLflow (Optional)

For experiment tracking:

```bash
# Start MLflow UI
mlflow ui --port 5000
```

MLflow will be available at `http://localhost:5000`

## 🔧 Configuration

### Environment Variables

Create a `.env.local` file in the `quantum-portfolio-web` directory:

```env
# API Configuration
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws

# MLflow Configuration (optional)
NEXT_PUBLIC_MLFLOW_URL=http://localhost:5000

# Development
NODE_ENV=development
```

### Backend Configuration

The backend server (`backend/server.py`) provides mock data and API endpoints. For production use, replace with actual QEPO backend integration.

## 📊 Features Overview

### 1. Data Ingestion
- **Sources**: Wikipedia (S&P 500), Kaggle, yfinance
- **Configuration**: Date ranges, caching options
- **Real-time**: Progress tracking and status updates

### 2. Portfolio Optimization
- **Algorithms**: QAOA (quantum), MVO, Greedy
- **Parameters**: Risk aversion, cardinality, weight bounds
- **QAOA Settings**: Circuit depth, shots, optimizer
- **Hardware**: IBM Quantum integration options

### 3. Backtesting
- **Strategies**: Multiple algorithm comparison
- **Analysis**: Walk-forward, rolling window
- **Metrics**: Sharpe ratio, drawdown, turnover
- **Benchmarks**: SPY, custom benchmarks

### 4. Reporting
- **MLflow Integration**: Experiment tracking
- **Visualizations**: Equity curves, allocations, metrics
- **Export**: HTML, PDF, Markdown formats
- **Performance**: Attribution analysis

### 5. Configuration Management
- **YAML Editor**: Data, optimizer, backtest configs
- **Validation**: Real-time error checking
- **Templates**: Pre-configured settings
- **Import/Export**: Configuration sharing

### 6. Job Monitoring
- **Real-time**: WebSocket updates
- **Status**: Running, completed, failed
- **Logs**: Detailed execution logs
- **Management**: Cancel, delete jobs

## 🎨 User Interface

### Navigation
- **Tab-based**: Easy switching between features
- **Responsive**: Mobile-friendly design
- **Status Indicators**: Real-time job status
- **Progress Bars**: Visual progress tracking

### Components
- **Forms**: Validated input with real-time feedback
- **Charts**: Interactive visualizations with Recharts
- **Tables**: Sortable data tables
- **Modals**: Detailed views and confirmations

### Styling
- **Tailwind CSS**: Utility-first styling
- **Quantum Theme**: Custom color scheme
- **Icons**: Heroicons for consistent iconography
- **Animations**: Smooth transitions and loading states

## 🔌 API Integration

### REST Endpoints
- `POST /data` - Start data ingestion
- `POST /optimize` - Start portfolio optimization
- `POST /backtest` - Start backtesting
- `POST /report/generate` - Generate reports
- `GET /mlflow/runs` - Get experiment runs
- `GET /jobs` - Get job status
- `GET /configs` - Get configurations

### WebSocket
- Real-time job updates
- Progress notifications
- Status changes
- Error messages

## 🚀 Deployment

### Development
```bash
npm run dev          # Start development server
npm run build        # Build for production
npm run start        # Start production server
npm run lint         # Run ESLint
npm run type-check   # Run TypeScript checks
```

### Production
```bash
# Build the application
npm run build

# Start production server
npm start

# Or use PM2 for process management
pm2 start npm --name "qepo-web" -- start
```

### Docker
```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

## 🐛 Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Kill process on port 3000
   npx kill-port 3000
   
   # Or use different port
   npm run dev -- -p 3001
   ```

2. **API Connection Failed**
   - Check if backend server is running on port 8000
   - Verify environment variables in `.env.local`
   - Check CORS settings in backend

3. **MLflow Connection Issues**
   - Ensure MLflow server is running on port 5000
   - Check MLflow URL in environment variables
   - Verify MLflow backend storage

4. **Build Errors**
   ```bash
   # Clear cache and reinstall
   rm -rf node_modules package-lock.json
   npm install
   npm run build
   ```

### Debug Mode
```bash
# Enable debug logging
DEBUG=* npm run dev

# Or set environment variable
NODE_ENV=development npm run dev
```

## 📚 Additional Resources

### Documentation
- [Next.js Documentation](https://nextjs.org/docs)
- [Tailwind CSS Documentation](https://tailwindcss.com/docs)
- [React Query Documentation](https://tanstack.com/query/latest)
- [Recharts Documentation](https://recharts.org/)

### QEPO Backend
- [QEPO Documentation](../quantum-portfolio/README.md)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Qiskit Documentation](https://qiskit.org/documentation/)

### Support
- Check the main README.md for detailed information
- Review component documentation in the code
- Open issues on GitHub for bugs or feature requests

## 🎯 Next Steps

1. **Explore the Interface**: Navigate through all tabs and features
2. **Run Sample Jobs**: Try data ingestion, optimization, and backtesting
3. **Customize Configurations**: Modify settings for your use case
4. **Generate Reports**: Create and download performance reports
5. **Monitor Jobs**: Use the job monitor for real-time updates

## 🔮 Future Enhancements

- **User Authentication**: Multi-user support with permissions
- **Advanced Visualizations**: 3D portfolio visualization
- **Real-time Trading**: Live portfolio management
- **Mobile App**: React Native mobile application
- **Plugin System**: Extensible architecture for custom algorithms

---

**Happy optimizing! 🚀**
