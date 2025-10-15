'use client'

import { useState } from 'react'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'
import toast from 'react-hot-toast'
import { 
  BeakerIcon, 
  ChartBarIcon,
  ClockIcon,
  PlayIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  ArrowTrendingUpIcon,
  ArrowTrendingDownIcon
} from '@heroicons/react/24/outline'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts'
import { QuantumLoadingScreen, useQuantumLoading } from './QuantumLoadingScreen'

const backtestSchema = z.object({
  strategy: z.enum(['qaoa', 'mvo', 'greedy']),
  rebalanceFrequency: z.enum(['daily', 'weekly', 'monthly', 'quarterly']),
  trainMonths: z.number().min(1).max(60),
  testMonths: z.number().min(1).max(24),
  transactionCostsBps: z.number().min(0).max(100),
  benchmarkTicker: z.string().min(1),
  rollingWindow: z.boolean(),
  // Strategy-specific parameters
  riskAversion: z.number().min(0).max(10),
  cardinalityK: z.number().min(1).max(100),
  weightBounds: z.object({
    min: z.number().min(0).max(1),
    max: z.number().min(0).max(1),
  }),
})

type BacktestForm = z.infer<typeof backtestSchema>

interface BacktestJob {
  id: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  startTime: string
  endTime?: string
  progress: number
  message: string
  results?: {
    totalReturn: number
    annualizedReturn: number
    volatility: number
    sharpeRatio: number
    maxDrawdown: number
    avgTurnover: number
    benchmarkReturn: number
    benchmarkVolatility: number
    benchmarkSharpe: number
    equityCurve: Array<{ date: string; portfolio: number; benchmark: number }>
    monthlyReturns: Array<{ month: string; portfolio: number; benchmark: number }>
    drawdowns: Array<{ date: string; drawdown: number }>
  }
}

export function Backtesting() {
  const [isLoading, setIsLoading] = useState(false)
  const [currentJob, setCurrentJob] = useState<BacktestJob | null>(null)
  const [activeTab, setActiveTab] = useState<'config' | 'results'>('config')
  
  // Quantum loading hook
  const quantumLoading = useQuantumLoading()
  
  const {
    register,
    handleSubmit,
    formState: { errors },
    watch,
  } = useForm<BacktestForm>({
    resolver: zodResolver(backtestSchema),
    defaultValues: {
      strategy: 'qaoa',
      rebalanceFrequency: 'monthly',
      trainMonths: 36,
      testMonths: 12,
      transactionCostsBps: 5,
      benchmarkTicker: 'SPY',
      rollingWindow: true,
      riskAversion: 0.5,
      cardinalityK: 25,
      weightBounds: { min: 0.0, max: 0.1 },
    },
  })

  const strategy = watch('strategy')

  const onSubmit = async (data: BacktestForm) => {
    setIsLoading(true)
    setActiveTab('results')
    
    // Start quantum loading for backtesting
    quantumLoading.startLoading('backtest', 'Initializing backtest...')
    
    const job: BacktestJob = {
      id: `backtest-${Date.now()}`,
      status: 'running',
      startTime: new Date().toISOString(),
      progress: 0,
      message: 'Initializing backtest...',
    }
    
    setCurrentJob(job)
    
    try {
      // Simulate API call to QEPO backend
      const response = await fetch('/api/qepo/backtest', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          strategy: data.strategy,
          rebalance: data.rebalanceFrequency,
          window: {
            train_months: data.trainMonths,
            test_months: data.testMonths,
          },
          costs_bps: data.transactionCostsBps,
          benchmark: data.benchmarkTicker,
          rolling: data.rollingWindow,
          constraints: {
            cardinality_k: data.cardinalityK,
            weight_bounds: [data.weightBounds.min, data.weightBounds.max],
          },
          objective: {
            risk_aversion: data.riskAversion,
          },
        }),
      })

      if (!response.ok) {
        throw new Error('Failed to start backtest')
      }

      const result = await response.json()
      
      // Start polling for job status
      const pollJobStatus = async () => {
        try {
          const statusResponse = await fetch(`/api/qepo/backtest/status/${result.job_id}`)
          const statusData = await statusResponse.json()
          
          // Update quantum loading progress
          quantumLoading.updateProgress(statusData.progress, statusData.message)
          
          setCurrentJob({
            id: statusData.id,
            status: statusData.status,
            startTime: statusData.startTime,
            endTime: statusData.endTime,
            progress: statusData.progress,
            message: statusData.message,
            results: statusData.results ? {
              totalReturn: statusData.results.total_return || 0,
              annualizedReturn: statusData.results.annualized_return || 0,
              volatility: statusData.results.volatility || 0,
              sharpeRatio: statusData.results.sharpe_ratio || 0,
              maxDrawdown: statusData.results.max_drawdown || 0,
              avgTurnover: statusData.results.avg_turnover || 0,
              benchmarkReturn: statusData.results.benchmark_return || 0,
              benchmarkVolatility: statusData.results.benchmark_volatility || 0,
              benchmarkSharpe: statusData.results.benchmark_sharpe || 0,
              equityCurve: statusData.results.equity_curve || [],
              monthlyReturns: statusData.results.monthly_returns || [],
              drawdowns: statusData.results.drawdowns || [],
            } : undefined,
          })
          
          if (statusData.status === 'running') {
            setTimeout(pollJobStatus, 2000) // Poll every 2 seconds
          } else if (statusData.status === 'completed') {
            quantumLoading.stopLoading()
            toast.success('Backtest completed successfully!')
          } else if (statusData.status === 'failed') {
            quantumLoading.stopLoading()
            toast.error('Backtest failed')
          }
        } catch (error) {
          console.error('Error polling job status:', error)
          quantumLoading.stopLoading()
          toast.error('Failed to get job status')
        }
      }
      
      // Start polling
      setTimeout(pollJobStatus, 1000)
      
      toast.success('Backtest started!')
    } catch (error) {
      quantumLoading.stopLoading()
      setCurrentJob({
        ...job,
        status: 'failed',
        endTime: new Date().toISOString(),
        message: `Backtest failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
      })
      
      toast.error('Backtest failed')
    } finally {
      setIsLoading(false)
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircleIcon className="h-5 w-5 text-green-500" />
      case 'failed':
        return <ExclamationTriangleIcon className="h-5 w-5 text-red-500" />
      case 'running':
        return <div className="quantum-spinner" />
      default:
        return <ClockIcon className="h-5 w-5 text-gray-400" />
    }
  }

  return (
    <div className="space-y-6">
      <div className="text-center">
        <BeakerIcon className="mx-auto h-12 w-12 text-quantum-500" />
        <h2 className="mt-2 text-3xl font-bold text-gray-900">Backtesting</h2>
        <p className="mt-2 text-sm text-gray-600">
          Test your portfolio strategies against historical data
        </p>
      </div>

      {/* Tab Navigation */}
      <div className="border-b border-gray-200">
        <nav className="-mb-px flex space-x-8">
          <button
            onClick={() => setActiveTab('config')}
            className={`py-2 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'config'
                ? 'border-quantum-500 text-quantum-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            <BeakerIcon className="h-4 w-4 inline mr-2" />
            Configuration
          </button>
          <button
            onClick={() => setActiveTab('results')}
            className={`py-2 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'results'
                ? 'border-quantum-500 text-quantum-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            <ChartBarIcon className="h-4 w-4 inline mr-2" />
            Results
          </button>
        </nav>
      </div>

      {activeTab === 'config' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Configuration Form */}
          <div className="card">
            <div className="card-header">
              <h3 className="text-lg font-medium text-gray-900">Backtest Settings</h3>
            </div>
            <div className="card-body">
              <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
                {/* Strategy Selection */}
                <div>
                  <label className="form-label">Strategy</label>
                  <div className="mt-2 space-y-2">
                    <label className="flex items-center">
                      <input
                        type="radio"
                        value="qaoa"
                        {...register('strategy')}
                        className="h-4 w-4 text-quantum-600 focus:ring-quantum-500 border-gray-300"
                      />
                      <span className="ml-2 text-sm text-gray-900">
                        <span className="font-medium">QAOA</span> - Quantum Portfolio Optimization
                      </span>
                    </label>
                    <label className="flex items-center">
                      <input
                        type="radio"
                        value="mvo"
                        {...register('strategy')}
                        className="h-4 w-4 text-quantum-600 focus:ring-quantum-500 border-gray-300"
                      />
                      <span className="ml-2 text-sm text-gray-900">
                        <span className="font-medium">MVO</span> - Mean-Variance Optimization
                      </span>
                    </label>
                    <label className="flex items-center">
                      <input
                        type="radio"
                        value="greedy"
                        {...register('strategy')}
                        className="h-4 w-4 text-quantum-600 focus:ring-quantum-500 border-gray-300"
                      />
                      <span className="ml-2 text-sm text-gray-900">
                        <span className="font-medium">Greedy</span> - Greedy K-Selection
                      </span>
                    </label>
                  </div>
                </div>

                {/* Rebalancing */}
                <div>
                  <label className="form-label">Rebalancing Frequency</label>
                  <select {...register('rebalanceFrequency')} className="form-input">
                    <option value="daily">Daily</option>
                    <option value="weekly">Weekly</option>
                    <option value="monthly">Monthly</option>
                    <option value="quarterly">Quarterly</option>
                  </select>
                </div>

                {/* Time Windows */}
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="form-label">Training Period (months)</label>
                    <input
                      type="number"
                      min="1"
                      max="60"
                      {...register('trainMonths', { valueAsNumber: true })}
                      className="form-input"
                    />
                    {errors.trainMonths && (
                      <p className="mt-1 text-sm text-red-600">{errors.trainMonths.message}</p>
                    )}
                  </div>
                  <div>
                    <label className="form-label">Testing Period (months)</label>
                    <input
                      type="number"
                      min="1"
                      max="24"
                      {...register('testMonths', { valueAsNumber: true })}
                      className="form-input"
                    />
                    {errors.testMonths && (
                      <p className="mt-1 text-sm text-red-600">{errors.testMonths.message}</p>
                    )}
                  </div>
                </div>

                {/* Transaction Costs */}
                <div>
                  <label className="form-label">Transaction Costs (bps)</label>
                  <input
                    type="number"
                    min="0"
                    max="100"
                    step="0.1"
                    {...register('transactionCostsBps', { valueAsNumber: true })}
                    className="form-input"
                  />
                </div>

                {/* Benchmark */}
                <div>
                  <label className="form-label">Benchmark Ticker</label>
                  <input
                    type="text"
                    {...register('benchmarkTicker')}
                    className="form-input"
                    placeholder="SPY"
                  />
                </div>

                {/* Rolling Window */}
                <div className="flex items-center">
                  <input
                    type="checkbox"
                    {...register('rollingWindow')}
                    className="h-4 w-4 text-quantum-600 focus:ring-quantum-500 border-gray-300 rounded"
                  />
                  <label className="ml-2 block text-sm text-gray-900">
                    Use rolling window (walk-forward analysis)
                  </label>
                </div>

                {/* Strategy Parameters */}
                <div className="border-t pt-4 space-y-4">
                  <h4 className="text-md font-medium text-gray-900">Strategy Parameters</h4>
                  
                  <div>
                    <label className="form-label">Risk Aversion (λ)</label>
                    <input
                      type="range"
                      min="0"
                      max="10"
                      step="0.1"
                      {...register('riskAversion', { valueAsNumber: true })}
                      className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                    />
                    <div className="flex justify-between text-xs text-gray-500 mt-1">
                      <span>Conservative (0)</span>
                      <span className="font-medium">{watch('riskAversion')}</span>
                      <span>Aggressive (10)</span>
                    </div>
                  </div>

                  <div>
                    <label className="form-label">Portfolio Size (K)</label>
                    <input
                      type="number"
                      min="1"
                      max="100"
                      {...register('cardinalityK', { valueAsNumber: true })}
                      className="form-input"
                    />
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="form-label">Min Weight</label>
                      <input
                        type="number"
                        min="0"
                        max="1"
                        step="0.01"
                        {...register('weightBounds.min', { valueAsNumber: true })}
                        className="form-input"
                      />
                    </div>
                    <div>
                      <label className="form-label">Max Weight</label>
                      <input
                        type="number"
                        min="0"
                        max="1"
                        step="0.01"
                        {...register('weightBounds.max', { valueAsNumber: true })}
                        className="form-input"
                      />
                    </div>
                  </div>
                </div>

                <button
                  type="submit"
                  disabled={isLoading}
                  className="btn-quantum w-full"
                >
                  {isLoading ? (
                    <>
                      <div className="quantum-spinner mr-2" />
                      Running Backtest...
                    </>
                  ) : (
                    <>
                      <PlayIcon className="h-4 w-4 mr-2" />
                      Start Backtest
                    </>
                  )}
                </button>
              </form>
            </div>
          </div>

          {/* Job Status */}
          <div className="space-y-6">
            {currentJob && (
              <div className="card">
                <div className="card-header">
                  <h3 className="text-lg font-medium text-gray-900">Job Status</h3>
                </div>
                <div className="card-body">
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center">
                        {getStatusIcon(currentJob.status)}
                        <span className="ml-2 text-sm font-medium text-gray-900">
                          Job {currentJob.id}
                        </span>
                      </div>
                      <span className={`status-${currentJob.status}`}>
                        {currentJob.status}
                      </span>
                    </div>

                    <div>
                      <div className="flex justify-between text-sm text-gray-600 mb-1">
                        <span>Progress</span>
                        <span>{currentJob.progress}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div
                          className="bg-quantum-600 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${currentJob.progress}%` }}
                        />
                      </div>
                    </div>

                    <div className="bg-gray-50 p-3 rounded-md">
                      <p className="text-sm text-gray-700">{currentJob.message}</p>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Backtest Information */}
            <div className="card">
              <div className="card-header">
                <h3 className="text-lg font-medium text-gray-900">Backtest Information</h3>
              </div>
              <div className="card-body">
                <div className="space-y-3">
                  <div className="flex items-center">
                    <ChartBarIcon className="h-5 w-5 text-quantum-500 mr-2" />
                    <span className="font-medium">Walk-Forward Analysis</span>
                  </div>
                  <p className="text-sm text-gray-600">
                    The backtest uses a walk-forward approach where the strategy is trained on historical 
                    data and then tested on out-of-sample data. This provides a more realistic assessment 
                    of strategy performance.
                  </p>
                  <ul className="text-sm text-gray-600 space-y-1">
                    <li>• Training period: {watch('trainMonths')} months</li>
                    <li>• Testing period: {watch('testMonths')} months</li>
                    <li>• Rebalancing: {watch('rebalanceFrequency')}</li>
                    <li>• Transaction costs: {watch('transactionCostsBps')} bps</li>
                    <li>• Benchmark: {watch('benchmarkTicker')}</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'results' && currentJob?.results && (
        <div className="space-y-6">
          {/* Performance Summary */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="metric-card text-center">
              <div className="metric-value text-emerald-600">
                {((currentJob.results.totalReturn || 0) * 100).toFixed(1)}%
              </div>
              <div className="metric-label">Total Return</div>
            </div>
            <div className="metric-card text-center">
              <div className="metric-value text-blue-600">
                {((currentJob.results.annualizedReturn || 0) * 100).toFixed(1)}%
              </div>
              <div className="metric-label">Annualized Return</div>
            </div>
            <div className="metric-card text-center">
              <div className="metric-value text-purple-600">
                {(currentJob.results.sharpeRatio || 0).toFixed(2)}
              </div>
              <div className="metric-label">Sharpe Ratio</div>
            </div>
            <div className="metric-card text-center">
              <div className="metric-value text-red-600">
                {((currentJob.results.maxDrawdown || 0) * 100).toFixed(1)}%
              </div>
              <div className="metric-label">Max Drawdown</div>
            </div>
          </div>

          {/* Portfolio vs Benchmark Comparison */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="card">
              <div className="card-header">
                <h3 className="text-lg font-medium text-gray-900">Portfolio Performance</h3>
              </div>
              <div className="card-body">
                <div className="space-y-4">
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Total Return</span>
                    <span className="font-medium">
                      {((currentJob.results.totalReturn || 0) * 100).toFixed(2)}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Annualized Return</span>
                    <span className="font-medium">
                      {((currentJob.results.annualizedReturn || 0) * 100).toFixed(2)}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Volatility</span>
                    <span className="font-medium">
                      {((currentJob.results.volatility || 0) * 100).toFixed(2)}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Sharpe Ratio</span>
                    <span className="font-medium">
                      {(currentJob.results.sharpeRatio || 0).toFixed(3)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Max Drawdown</span>
                    <span className="font-medium text-red-600">
                      {((currentJob.results.maxDrawdown || 0) * 100).toFixed(2)}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Avg Turnover</span>
                    <span className="font-medium">
                      {((currentJob.results.avgTurnover || 0) * 100).toFixed(2)}%
                    </span>
                  </div>
                </div>
              </div>
            </div>

            <div className="card">
              <div className="card-header">
                <h3 className="text-lg font-medium text-gray-900">Benchmark Performance</h3>
              </div>
              <div className="card-body">
                <div className="space-y-4">
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Total Return</span>
                    <span className="font-medium">
                      {((currentJob.results.benchmarkReturn || 0) * 100).toFixed(2)}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Annualized Return</span>
                    <span className="font-medium">
                      {((currentJob.results.benchmarkReturn || 0) * 100).toFixed(2)}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Volatility</span>
                    <span className="font-medium">
                      {((currentJob.results.benchmarkVolatility || 0) * 100).toFixed(2)}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Sharpe Ratio</span>
                    <span className="font-medium">
                      {(currentJob.results.benchmarkSharpe || 0).toFixed(3)}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Equity Curve Chart */}
          <div className="card">
            <div className="card-header">
              <h3 className="text-lg font-medium text-gray-900">Equity Curve</h3>
            </div>
            <div className="card-body">
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={currentJob.results.equityCurve}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis />
                    <Tooltip 
                      formatter={(value: number) => [`${(value * 100).toFixed(2)}%`, '']}
                      labelFormatter={(label) => `Date: ${label}`}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="portfolio" 
                      stroke="#0ea5e9" 
                      strokeWidth={2}
                      name="Portfolio"
                    />
                    <Line 
                      type="monotone" 
                      dataKey="benchmark" 
                      stroke="#6b7280" 
                      strokeWidth={2}
                      name="Benchmark"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* Monthly Returns */}
          <div className="card">
            <div className="card-header">
              <h3 className="text-lg font-medium text-gray-900">Monthly Returns</h3>
            </div>
            <div className="card-body">
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={currentJob.results.monthlyReturns}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="month" />
                    <YAxis />
                    <Tooltip 
                      formatter={(value: number) => [`${(value * 100).toFixed(2)}%`, '']}
                    />
                    <Bar dataKey="portfolio" fill="#0ea5e9" name="Portfolio" />
                    <Bar dataKey="benchmark" fill="#6b7280" name="Benchmark" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        </div>
      )}
      
      {/* Quantum Loading Screen */}
      <QuantumLoadingScreen
        isLoading={quantumLoading.isLoading}
        progress={quantumLoading.progress}
        message={quantumLoading.message}
        algorithm={quantumLoading.algorithm}
        onCancel={() => {
          quantumLoading.stopLoading()
          setIsLoading(false)
        }}
      />
    </div>
  )
}
