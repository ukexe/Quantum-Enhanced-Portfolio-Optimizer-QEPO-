# Quantum Portfolio Optimizer - Web Interface

A comprehensive web interface for the Quantum-Enhanced Portfolio Optimizer (QEPO), providing an intuitive way to interact with quantum portfolio optimization algorithms through a modern React-based dashboard.

## 🚀 Features

### Core Functionality
- **Data Ingestion**: Download and manage S&P 500 market data from multiple sources
- **Portfolio Optimization**: Run quantum (QAOA) and classical (MVO, Greedy) optimization algorithms
- **Backtesting**: Comprehensive walk-forward analysis with multiple strategies
- **Reporting**: Generate detailed reports with visualizations and performance metrics
- **Configuration Management**: Manage YAML configuration files for all components
- **Real-time Monitoring**: Live job tracking and status updates

### Technical Features
- **Modern UI**: Built with Next.js 14, TypeScript, and Tailwind CSS
- **Real-time Updates**: WebSocket integration for live job monitoring
- **MLflow Integration**: Complete experiment tracking and artifact management
- **Responsive Design**: Mobile-friendly interface with adaptive layouts
- **Interactive Charts**: Rich visualizations using Recharts
- **Form Validation**: Comprehensive form validation with Zod schemas

## 🛠️ Installation

### Prerequisites
- Node.js 18+ 
- Python 3.11+ (for QEPO backend)
- MLflow server running on port 5000 (see [MLflow Setup Guide](./MLFLOW_SETUP.md))

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd quantum-portfolio-web
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Start MLflow Server** (Required for Reporting & Analytics)
   ```bash
   cd backend
   python start_mlflow.py
   ```
   See [MLflow Setup Guide](./MLFLOW_SETUP.md) for detailed instructions.

4. **Environment Configuration**
   Create a `.env.local` file:
   ```env
   NEXT_PUBLIC_API_URL=http://localhost:8000
   NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws
   NEXT_PUBLIC_MLFLOW_URL=http://localhost:5000
   ```

5. **Start the development server**
   ```bash
   npm run dev
   ```

6. **Start the QEPO backend** (Optional - for running actual optimizations)
   ```bash
   cd ../quantum-portfolio
   python -m qepo.cli data ingest
   python -m qepo.cli optimize
   ```

7. **Start MLflow UI** (Alternative to step 3)
   ```bash
   mlflow ui --port 5000
   ```

The application will be available at `http://localhost:3000`.

## 📊 Usage

### Data Ingestion
1. Navigate to the **Data Ingestion** tab
2. Configure data sources (Wikipedia, Kaggle, yfinance)
3. Set date ranges and caching options
4. Click "Start Data Ingestion" to download market data

### Portfolio Optimization
1. Go to the **Portfolio Optimization** tab
2. Choose between QAOA (quantum) or classical algorithms
3. Configure risk parameters, constraints, and solver settings
4. For QAOA: Set circuit depth, shots, and optimizer parameters
5. Click "Start Optimization" to run the algorithm

### Backtesting
1. Access the **Backtesting** tab
2. Select strategy and rebalancing frequency
3. Configure training/testing periods and transaction costs
4. Choose benchmarks and performance metrics
5. Run backtest to analyze historical performance

### Reporting
1. Visit the **Reporting** tab
2. Select an MLflow run from the list
3. View performance metrics and visualizations
4. Generate reports in HTML, Markdown, or PDF format

### Configuration Management
1. Use the **Configuration** tab to manage settings
2. Edit data, optimizer, and backtest configurations
3. Validate configurations before saving
4. Load existing configurations for reuse

### Job Monitoring
1. Monitor all active jobs in the **Job Monitor** tab
2. View real-time progress and status updates
3. Access detailed logs and configuration details
4. Cancel or delete jobs as needed

## 🏗️ Architecture

### Frontend Stack
- **Next.js 14**: React framework with App Router
- **TypeScript**: Type-safe development
- **Tailwind CSS**: Utility-first CSS framework
- **React Query**: Server state management
- **React Hook Form**: Form handling with validation
- **Zod**: Schema validation
- **Recharts**: Data visualization
- **Heroicons**: Icon library

### Backend Integration
- **REST API**: HTTP endpoints for all operations
- **WebSocket**: Real-time job monitoring
- **MLflow**: Experiment tracking and artifact storage
- **File System**: Configuration and data management

### Key Components

#### Navigation
- Tab-based navigation with responsive design
- Active state management and mobile support

#### Data Ingestion
- Multi-source data configuration
- Progress tracking and error handling
- Caching options and validation

#### Portfolio Optimization
- Algorithm selection (QAOA, MVO, Greedy)
- Parameter configuration with real-time validation
- Hardware integration options
- Results visualization

#### Backtesting
- Walk-forward analysis configuration
- Multiple strategy comparison
- Performance metrics and charts
- Benchmark analysis

#### Reporting
- MLflow integration for experiment data
- Interactive charts and visualizations
- Export functionality (HTML, PDF, Markdown)
- Performance attribution analysis

#### Configuration Management
- YAML configuration editing
- Validation and error reporting
- Template management
- Import/export functionality

#### Job Monitor
- Real-time job status updates
- Progress tracking and logging
- Job management (cancel, delete)
- WebSocket integration

## 🔧 Configuration

### Environment Variables
```env
# API Configuration
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws

# MLflow Configuration
NEXT_PUBLIC_MLFLOW_URL=http://localhost:5000

# Development
NODE_ENV=development
```

### API Endpoints
The web interface expects the following backend endpoints:

- `POST /api/qepo/data` - Data ingestion
- `POST /api/qepo/optimize` - Portfolio optimization
- `POST /api/qepo/backtest` - Backtesting
- `POST /api/qepo/report/generate` - Report generation
- `GET /api/qepo/mlflow/runs` - MLflow runs
- `GET /api/qepo/jobs` - Job monitoring
- `GET /api/qepo/configs` - Configuration management

## 🎨 Customization

### Styling
The interface uses Tailwind CSS with custom quantum-themed colors:
- Primary: Blue tones for main actions
- Quantum: Cyan tones for quantum-specific features
- Status colors for job states and metrics

### Components
All components are modular and can be customized:
- Form components with validation
- Chart components with Recharts
- Status indicators and progress bars
- Modal dialogs and notifications

### Themes
The interface supports light/dark mode switching (can be extended):
- Light mode (default)
- Dark mode (future enhancement)

## 🚀 Deployment

### Production Build
```bash
npm run build
npm start
```

### Docker Deployment
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

### Environment Setup
For production deployment:
1. Set up reverse proxy (nginx)
2. Configure SSL certificates
3. Set up monitoring and logging
4. Configure MLflow backend storage
5. Set up job queue system

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Development Guidelines
- Use TypeScript for all new code
- Follow the existing component structure
- Add proper error handling
- Include loading states
- Write responsive designs
- Add proper accessibility attributes

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Check the documentation
- Review the QEPO backend documentation
- Open an issue on GitHub
- Contact the development team

## 🔮 Future Enhancements

- **Advanced Visualizations**: 3D portfolio visualization
- **Real-time Trading**: Live portfolio management
- **Multi-user Support**: User authentication and permissions
- **Advanced Analytics**: Risk attribution and factor analysis
- **Mobile App**: React Native mobile application
- **API Documentation**: Interactive API documentation
- **Plugin System**: Extensible architecture for custom algorithms
