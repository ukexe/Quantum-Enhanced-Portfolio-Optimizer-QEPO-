'use client'

import { useState } from 'react'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'
import toast from 'react-hot-toast'
import { 
  CpuChipIcon, 
  BeakerIcon,
  ChartBarIcon,
  CogIcon,
  PlayIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  ClockIcon
} from '@heroicons/react/24/outline'
import { QuantumLoadingScreen, useQuantumLoading } from './QuantumLoadingScreen'

const optimizationSchema = z.object({
  solverType: z.enum(['qaoa', 'mvo', 'greedy']),
  riskAversion: z.number().min(0).max(10),
  cardinalityK: z.number().min(1).max(100),
  weightBounds: z.object({
    min: z.number().min(0).max(1),
    max: z.number().min(0).max(1),
  }),
  noShort: z.boolean(),
  transactionCostBps: z.number().min(0).max(100),
  // QAOA specific
  qaoaDepth: z.number().min(1).max(10).optional(),
  qaoaShots: z.number().min(100).max(10000).optional(),
  qaoaOptimizer: z.enum(['COBYLA', 'SPSA', 'L_BFGS_B']).optional(),
  // Hardware options
  useHardware: z.boolean(),
  hardwareBackend: z.string().optional(),
})

type OptimizationForm = z.infer<typeof optimizationSchema>

interface OptimizationJob {
  id: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  startTime: string
  endTime?: string
  progress: number
  message: string
  results?: {
    portfolioReturn: number
    portfolioVolatility: number
    sharpeRatio: number
    selectedAssets: number
    maxWeight: number
    topHoldings: Array<{ ticker: string; weight: number }>
  }
}

export function PortfolioOptimization() {
  const [isLoading, setIsLoading] = useState(false)
  const [currentJob, setCurrentJob] = useState<OptimizationJob | null>(null)
  const [activeTab, setActiveTab] = useState<'config' | 'results'>('config')
  
  // Quantum loading hook
  const quantumLoading = useQuantumLoading()
  
  const {
    register,
    handleSubmit,
    formState: { errors },
    watch,
    setValue,
  } = useForm<OptimizationForm>({
    resolver: zodResolver(optimizationSchema),
    defaultValues: {
      solverType: 'qaoa',
      riskAversion: 0.5,
      cardinalityK: 25,
      weightBounds: { min: 0.0, max: 0.1 },
      noShort: true,
      transactionCostBps: 5,
      qaoaDepth: 3,
      qaoaShots: 4096,
      qaoaOptimizer: 'SPSA',
      useHardware: false,
      hardwareBackend: '',
    },
  })

  const solverType = watch('solverType')
  const useHardware = watch('useHardware')

  const onSubmit = async (data: OptimizationForm) => {
    setIsLoading(true)
    setActiveTab('results')
    
    // Start quantum loading with appropriate algorithm type
    quantumLoading.startLoading(data.solverType, 'Initializing optimization...')
    
    const job: OptimizationJob = {
      id: `optimize-${Date.now()}`,
      status: 'running',
      startTime: new Date().toISOString(),
      progress: 0,
      message: 'Initializing optimization...',
    }
    
    setCurrentJob(job)
    
    try {
      // Simulate API call to QEPO backend
      const response = await fetch('/api/qepo/optimize', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          solver: {
            type: data.solverType,
            ...(data.solverType === 'qaoa' && {
              depth: data.qaoaDepth,
              shots: data.qaoaShots,
              optimizer: data.qaoaOptimizer,
            }),
          },
          objective: {
            risk_aversion: data.riskAversion,
            transaction_cost_bps: data.transactionCostBps,
          },
          constraints: {
            cardinality_k: data.cardinalityK,
            weight_bounds: [data.weightBounds.min, data.weightBounds.max],
            no_short: data.noShort,
          },
          hardware: {
            use_ibm: data.useHardware,
            backend_name: data.hardwareBackend,
          },
        }),
      })

      if (!response.ok) {
        throw new Error('Failed to start optimization')
      }

      const result = await response.json()
      
      // Start polling for job status
      const pollJobStatus = async () => {
        try {
          const statusResponse = await fetch(`/api/qepo/optimize/status/${result.job_id}`)
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
              portfolioReturn: statusData.results.portfolio_return || 0,
              portfolioVolatility: statusData.results.portfolio_volatility || 0,
              sharpeRatio: statusData.results.sharpe_ratio || 0,
              selectedAssets: statusData.results.num_selected_assets || 0,
              maxWeight: statusData.results.max_weight || 0,
              topHoldings: statusData.results.top_holdings || [],
            } : undefined,
          })
          
          if (statusData.status === 'running') {
            setTimeout(pollJobStatus, 2000) // Poll every 2 seconds
          } else if (statusData.status === 'completed') {
            quantumLoading.stopLoading()
            toast.success('Portfolio optimization completed successfully!')
          } else if (statusData.status === 'failed') {
            quantumLoading.stopLoading()
            toast.error('Portfolio optimization failed')
          }
        } catch (error) {
          console.error('Error polling job status:', error)
          quantumLoading.stopLoading()
          toast.error('Failed to get job status')
        }
      }
      
      // Start polling
      setTimeout(pollJobStatus, 1000)
      
      toast.success('Portfolio optimization started!')
    } catch (error) {
      quantumLoading.stopLoading()
      setCurrentJob({
        ...job,
        status: 'failed',
        endTime: new Date().toISOString(),
        message: `Optimization failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
      })
      
      toast.error('Portfolio optimization failed')
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
        <CpuChipIcon className="mx-auto h-12 w-12 text-quantum-500" />
        <h2 className="mt-2 text-3xl font-bold text-gray-900">Portfolio Optimization</h2>
        <p className="mt-2 text-sm text-gray-600">
          Optimize your portfolio using quantum or classical algorithms
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
            <CogIcon className="h-4 w-4 inline mr-2" />
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
              <h3 className="text-lg font-medium text-gray-900">Optimization Settings</h3>
            </div>
            <div className="card-body">
              <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
                {/* Solver Type */}
                <div>
                  <label className="form-label">Solver Type</label>
                  <div className="mt-2 space-y-2">
                    <label className="flex items-center">
                      <input
                        type="radio"
                        value="qaoa"
                        {...register('solverType')}
                        className="h-4 w-4 text-quantum-600 focus:ring-quantum-500 border-gray-300"
                      />
                      <span className="ml-2 text-sm text-gray-900">
                        <span className="font-medium">QAOA</span> - Quantum Approximate Optimization Algorithm
                      </span>
                    </label>
                    <label className="flex items-center">
                      <input
                        type="radio"
                        value="mvo"
                        {...register('solverType')}
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
                        {...register('solverType')}
                        className="h-4 w-4 text-quantum-600 focus:ring-quantum-500 border-gray-300"
                      />
                      <span className="ml-2 text-sm text-gray-900">
                        <span className="font-medium">Greedy</span> - Greedy K-selection
                      </span>
                    </label>
                  </div>
                </div>

                {/* Risk Aversion */}
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

                {/* Cardinality */}
                <div>
                  <label className="form-label">Portfolio Size (K)</label>
                  <input
                    type="number"
                    min="1"
                    max="100"
                    {...register('cardinalityK', { valueAsNumber: true })}
                    className="form-input"
                  />
                  {errors.cardinalityK && (
                    <p className="mt-1 text-sm text-red-600">{errors.cardinalityK.message}</p>
                  )}
                </div>

                {/* Weight Bounds */}
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

                {/* Transaction Costs */}
                <div>
                  <label className="form-label">Transaction Costs (bps)</label>
                  <input
                    type="number"
                    min="0"
                    max="100"
                    step="0.1"
                    {...register('transactionCostBps', { valueAsNumber: true })}
                    className="form-input"
                  />
                </div>

                {/* Constraints */}
                <div className="space-y-3">
                  <div className="flex items-center">
                    <input
                      type="checkbox"
                      {...register('noShort')}
                      className="h-4 w-4 text-quantum-600 focus:ring-quantum-500 border-gray-300 rounded"
                    />
                    <label className="ml-2 block text-sm text-gray-900">
                      No short selling
                    </label>
                  </div>
                </div>

                {/* QAOA Specific Settings */}
                {solverType === 'qaoa' && (
                  <div className="border-t pt-4 space-y-4">
                    <h4 className="text-md font-medium text-gray-900">QAOA Parameters</h4>
                    
                    <div>
                      <label className="form-label">Circuit Depth (p)</label>
                      <input
                        type="number"
                        min="1"
                        max="10"
                        {...register('qaoaDepth', { valueAsNumber: true })}
                        className="form-input"
                      />
                    </div>

                    <div>
                      <label className="form-label">Number of Shots</label>
                      <select {...register('qaoaShots', { valueAsNumber: true })} className="form-input">
                        <option value={1024}>1,024</option>
                        <option value={2048}>2,048</option>
                        <option value={4096}>4,096</option>
                        <option value={8192}>8,192</option>
                      </select>
                    </div>

                    <div>
                      <label className="form-label">Optimizer</label>
                      <select {...register('qaoaOptimizer')} className="form-input">
                        <option value="COBYLA">COBYLA</option>
                        <option value="SPSA">SPSA</option>
                        <option value="L_BFGS_B">L-BFGS-B</option>
                      </select>
                    </div>

                    <div className="flex items-center">
                      <input
                        type="checkbox"
                        {...register('useHardware')}
                        className="h-4 w-4 text-quantum-600 focus:ring-quantum-500 border-gray-300 rounded"
                      />
                      <label className="ml-2 block text-sm text-gray-900">
                        Use IBM Quantum Hardware
                      </label>
                    </div>

                    {useHardware && (
                      <div>
                        <label className="form-label">Hardware Backend</label>
                        <select {...register('hardwareBackend')} className="form-input">
                          <option value="">Select backend...</option>
                          <option value="ibmq_qasm_simulator">IBM QASM Simulator</option>
                          <option value="ibmq_lima">IBM Lima (5 qubits)</option>
                          <option value="ibmq_belem">IBM Belem (5 qubits)</option>
                          <option value="ibmq_quito">IBM Quito (5 qubits)</option>
                        </select>
                      </div>
                    )}
                  </div>
                )}

                <button
                  type="submit"
                  disabled={isLoading}
                  className="btn-quantum w-full"
                >
                  {isLoading ? (
                    <>
                      <div className="quantum-spinner mr-2" />
                      Optimizing Portfolio...
                    </>
                  ) : (
                    <>
                      <PlayIcon className="h-4 w-4 mr-2" />
                      Start Optimization
                    </>
                  )}
                </button>
              </form>
            </div>
          </div>

          {/* Algorithm Information */}
          <div className="space-y-6">
            <div className="card">
              <div className="card-header">
                <h3 className="text-lg font-medium text-gray-900">Algorithm Information</h3>
              </div>
              <div className="card-body">
                {solverType === 'qaoa' && (
                  <div className="space-y-3">
                    <div className="flex items-center">
                      <CpuChipIcon className="h-5 w-5 text-quantum-500 mr-2" />
                      <span className="font-medium">Quantum Approximate Optimization Algorithm</span>
                    </div>
                    <p className="text-sm text-gray-600">
                      QAOA is a quantum algorithm that uses quantum circuits to find approximate solutions 
                      to combinatorial optimization problems. It's particularly effective for portfolio 
                      optimization with complex constraints.
                    </p>
                    <ul className="text-sm text-gray-600 space-y-1">
                      <li>• Handles complex constraints naturally</li>
                      <li>• Can find better solutions than classical methods</li>
                      <li>• Suitable for medium-sized portfolios (10-50 assets)</li>
                      <li>• Requires quantum hardware or high-fidelity simulators</li>
                    </ul>
                  </div>
                )}
                
                {solverType === 'mvo' && (
                  <div className="space-y-3">
                    <div className="flex items-center">
                      <ChartBarIcon className="h-5 w-5 text-blue-500 mr-2" />
                      <span className="font-medium">Mean-Variance Optimization</span>
                    </div>
                    <p className="text-sm text-gray-600">
                      MVO is the classical approach to portfolio optimization, balancing expected return 
                      against risk (variance). It's fast and reliable for most portfolio optimization tasks.
                    </p>
                    <ul className="text-sm text-gray-600 space-y-1">
                      <li>• Fast and computationally efficient</li>
                      <li>• Well-established theoretical foundation</li>
                      <li>• Good for large portfolios (100+ assets)</li>
                      <li>• May struggle with complex constraints</li>
                    </ul>
                  </div>
                )}
                
                {solverType === 'greedy' && (
                  <div className="space-y-3">
                    <div className="flex items-center">
                      <BeakerIcon className="h-5 w-5 text-green-500 mr-2" />
                      <span className="font-medium">Greedy K-Selection</span>
                    </div>
                    <p className="text-sm text-gray-600">
                      A heuristic approach that iteratively selects the best assets based on a scoring 
                      function. It's simple but can be surprisingly effective for cardinality-constrained portfolios.
                    </p>
                    <ul className="text-sm text-gray-600 space-y-1">
                      <li>• Very fast execution</li>
                      <li>• Naturally handles cardinality constraints</li>
                      <li>• Good baseline for comparison</li>
                      <li>• May not find globally optimal solutions</li>
                    </ul>
                  </div>
                )}
              </div>
            </div>

            {/* Job Status */}
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
          </div>
        </div>
      )}

      {activeTab === 'results' && currentJob?.results && (
        <div className="space-y-6">
          {/* Performance Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="metric-card text-center">
              <div className="metric-value text-emerald-600">
                {((currentJob.results.portfolioReturn || 0) * 100).toFixed(2)}%
              </div>
              <div className="metric-label">Expected Return</div>
            </div>
            <div className="metric-card text-center">
              <div className="metric-value text-amber-600">
                {((currentJob.results.portfolioVolatility || 0) * 100).toFixed(2)}%
              </div>
              <div className="metric-label">Volatility</div>
            </div>
            <div className="metric-card text-center">
              <div className="metric-value text-blue-600">
                {(currentJob.results.sharpeRatio || 0).toFixed(3)}
              </div>
              <div className="metric-label">Sharpe Ratio</div>
            </div>
            <div className="metric-card text-center">
              <div className="metric-value text-purple-600">
                {currentJob.results.selectedAssets || 0}
              </div>
              <div className="metric-label">Selected Assets</div>
            </div>
          </div>

          {/* Top Holdings */}
          <div className="card">
            <div className="card-header">
              <h3 className="text-lg font-medium text-gray-900">Top Holdings</h3>
            </div>
            <div className="card-body">
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Ticker
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Weight
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Percentage
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {(currentJob.results.topHoldings || []).map((holding, index) => (
                      <tr key={index}>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                          {holding.ticker || 'N/A'}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {(holding.weight || 0).toFixed(4)}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {((holding.weight || 0) * 100).toFixed(2)}%
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
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
