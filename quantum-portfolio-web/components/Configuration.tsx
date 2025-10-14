'use client'

import { useState, useEffect } from 'react'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'
import toast from 'react-hot-toast'
import { 
  CogIcon, 
  DocumentIcon,
  ArrowDownTrayIcon,
  ArrowPathIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  InformationCircleIcon
} from '@heroicons/react/24/outline'

const configSchema = z.object({
  data: z.object({
    universe: z.object({
      source: z.array(z.string()),
      tickers: z.string().optional(),
    }),
    history: z.object({
      start: z.string(),
      end: z.string(),
    }),
    risk_free: z.object({
      source: z.string(),
      series: z.string(),
    }),
    cache: z.object({
      enable: z.boolean(),
      path: z.string(),
    }),
  }),
  optimizer: z.object({
    objective: z.object({
      risk_aversion: z.number().min(0).max(10),
      transaction_cost_bps: z.number().min(0).max(100),
    }),
    constraints: z.object({
      cardinality_k: z.number().min(1).max(100),
      weight_bounds: z.array(z.number()),
      sector_caps: z.record(z.number()).optional(),
      no_short: z.boolean(),
    }),
    qubo: z.object({
      penalty_mode: z.string(),
      encoding: z.string(),
    }),
    solver: z.object({
      type: z.string(),
      backend: z.string(),
      shots: z.number(),
      p_depth: z.number(),
      optimizer: z.string(),
      restarts: z.number(),
    }),
    hardware: z.object({
      use_ibm: z.boolean(),
      backend_name: z.string().optional(),
    }),
    seed: z.number(),
  }),
  backtest: z.object({
    rebalance: z.string(),
    costs_bps: z.number(),
    window: z.object({
      train_months: z.number(),
      test_months: z.number(),
    }),
    rolling: z.boolean(),
    benchmarks: z.array(z.string()),
    metrics: z.array(z.string()),
  }),
})

type ConfigForm = z.infer<typeof configSchema>

interface ConfigFile {
  name: string
  content: any
  lastModified: string
  isValid: boolean
  errors?: string[]
}

export function Configuration() {
  const [configs, setConfigs] = useState<ConfigFile[]>([])
  const [selectedConfig, setSelectedConfig] = useState<string>('')
  const [isLoading, setIsLoading] = useState(false)
  const [isSaving, setIsSaving] = useState(false)
  const [activeTab, setActiveTab] = useState<'data' | 'optimizer' | 'backtest'>('data')

  const {
    register,
    handleSubmit,
    formState: { errors },
    watch,
    setValue,
    reset,
  } = useForm<ConfigForm>({
    resolver: zodResolver(configSchema),
    defaultValues: {
      data: {
        universe: {
          source: ['wikipedia', 'kaggle'],
          tickers: '',
        },
        history: {
          start: '2018-01-01',
          end: '2025-01-01',
        },
        risk_free: {
          source: 'fred',
          series: 'DTB3',
        },
        cache: {
          enable: true,
          path: 'data/interim',
        },
      },
      optimizer: {
        objective: {
          risk_aversion: 0.5,
          transaction_cost_bps: 5,
        },
        constraints: {
          cardinality_k: 25,
          weight_bounds: [0.0, 0.1],
          sector_caps: { TECH: 0.35, FIN: 0.25 },
          no_short: true,
        },
        qubo: {
          penalty_mode: 'adaptive',
          encoding: 'binary_select',
        },
        solver: {
          type: 'qaoa',
          backend: 'aer_qasm',
          shots: 4096,
          p_depth: 3,
          optimizer: 'SPSA',
          restarts: 3,
        },
        hardware: {
          use_ibm: false,
          backend_name: '',
        },
        seed: 42,
      },
      backtest: {
        rebalance: 'monthly',
        costs_bps: 5,
        window: {
          train_months: 36,
          test_months: 12,
        },
        rolling: true,
        benchmarks: ['mvo', 'greedy_k'],
        metrics: ['sharpe', 'sortino', 'max_drawdown', 'turnover'],
      },
    },
  })

  // Load configurations on component mount
  useEffect(() => {
    loadConfigurations()
  }, [])

  const loadConfigurations = async () => {
    setIsLoading(true)
    try {
      const response = await fetch('/api/qepo/configs')
      if (!response.ok) throw new Error('Failed to load configurations')
      const configs = await response.json()
      setConfigs(configs)
    } catch (error) {
      toast.error('Failed to load configurations')
    } finally {
      setIsLoading(false)
    }
  }

  const loadConfig = async (configName: string) => {
    try {
      const response = await fetch(`/api/qepo/configs/${configName}`)
      if (!response.ok) throw new Error('Failed to load config')
      const config = await response.json()
      reset(config)
      setSelectedConfig(configName)
    } catch (error) {
      toast.error('Failed to load configuration')
    }
  }

  const saveConfig = async (data: ConfigForm) => {
    setIsSaving(true)
    try {
      const response = await fetch('/api/qepo/configs', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          name: selectedConfig || 'config',
          content: data,
        }),
      })

      if (!response.ok) throw new Error('Failed to save configuration')

      toast.success('Configuration saved successfully!')
      loadConfigurations()
    } catch (error) {
      toast.error('Failed to save configuration')
    } finally {
      setIsSaving(false)
    }
  }

  const validateConfig = (data: ConfigForm) => {
    try {
      configSchema.parse(data)
      return { isValid: true, errors: [] }
    } catch (error) {
      if (error instanceof z.ZodError) {
        return {
          isValid: false,
          errors: error.errors.map(err => `${err.path.join('.')}: ${err.message}`),
        }
      }
      return { isValid: false, errors: ['Unknown validation error'] }
    }
  }

  const getConfigIcon = (config: ConfigFile) => {
    if (config.isValid) {
      return <CheckCircleIcon className="h-4 w-4 text-green-500" />
    } else {
      return <ExclamationTriangleIcon className="h-4 w-4 text-red-500" />
    }
  }

  return (
    <div className="space-y-6">
      <div className="text-center">
        <CogIcon className="mx-auto h-12 w-12 text-quantum-500" />
        <h2 className="mt-2 text-3xl font-bold text-gray-900">Configuration Management</h2>
        <p className="mt-2 text-sm text-gray-600">
          Manage and validate configuration files for all QEPO components
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Configuration List */}
        <div className="lg:col-span-1">
          <div className="card">
            <div className="card-header">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-medium text-gray-900">Configurations</h3>
                <button
                  onClick={loadConfigurations}
                  className="text-quantum-600 hover:text-quantum-700"
                >
                  <ArrowPathIcon className="h-4 w-4" />
                </button>
              </div>
            </div>
            <div className="card-body">
              {isLoading ? (
                <div className="text-center py-4">
                  <div className="quantum-spinner mx-auto" />
                  <p className="text-sm text-gray-500 mt-2">Loading...</p>
                </div>
              ) : (
                <div className="space-y-2">
                  {configs.map((config) => (
                    <div
                      key={config.name}
                      onClick={() => loadConfig(config.name)}
                      className={`p-3 border rounded-lg cursor-pointer transition-colors ${
                        selectedConfig === config.name
                          ? 'border-quantum-500 bg-quantum-50'
                          : 'border-gray-200 hover:border-gray-300'
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center">
                          {getConfigIcon(config)}
                          <span className="ml-2 text-sm font-medium text-gray-900">
                            {config.name}
                          </span>
                        </div>
                      </div>
                      <div className="mt-1 text-xs text-gray-500">
                        {new Date(config.lastModified).toLocaleDateString()}
                      </div>
                      {!config.isValid && config.errors && (
                        <div className="mt-1 text-xs text-red-600">
                          {config.errors.length} error(s)
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Configuration Editor */}
        <div className="lg:col-span-3">
          <div className="card">
            <div className="card-header">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-medium text-gray-900">
                  {selectedConfig ? `Edit ${selectedConfig}` : 'Configuration Editor'}
                </h3>
                <button
                  onClick={handleSubmit(saveConfig)}
                  disabled={isSaving}
                  className="btn-quantum"
                >
                  {isSaving ? (
                    <>
                      <div className="quantum-spinner mr-2" />
                      Saving...
                    </>
                  ) : (
                    <>
                      <ArrowDownTrayIcon className="h-4 w-4 mr-2" />
                      Save
                    </>
                  )}
                </button>
              </div>
            </div>
            <div className="card-body">
              {/* Tab Navigation */}
              <div className="border-b border-gray-200 mb-6">
                <nav className="-mb-px flex space-x-8">
                  <button
                    onClick={() => setActiveTab('data')}
                    className={`py-2 px-1 border-b-2 font-medium text-sm ${
                      activeTab === 'data'
                        ? 'border-quantum-500 text-quantum-600'
                        : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                    }`}
                  >
                    Data Configuration
                  </button>
                  <button
                    onClick={() => setActiveTab('optimizer')}
                    className={`py-2 px-1 border-b-2 font-medium text-sm ${
                      activeTab === 'optimizer'
                        ? 'border-quantum-500 text-quantum-600'
                        : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                    }`}
                  >
                    Optimizer Configuration
                  </button>
                  <button
                    onClick={() => setActiveTab('backtest')}
                    className={`py-2 px-1 border-b-2 font-medium text-sm ${
                      activeTab === 'backtest'
                        ? 'border-quantum-500 text-quantum-600'
                        : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                    }`}
                  >
                    Backtest Configuration
                  </button>
                </nav>
              </div>

              <form onSubmit={handleSubmit(saveConfig)} className="space-y-6">
                {activeTab === 'data' && (
                  <div className="space-y-6">
                    <div>
                      <h4 className="text-md font-medium text-gray-900 mb-4">Data Sources</h4>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                          <label className="form-label">Universe Sources</label>
                          <div className="space-y-2">
                            <label className="flex items-center">
                              <input
                                type="checkbox"
                                {...register('data.universe.source')}
                                value="wikipedia"
                                className="h-4 w-4 text-quantum-600 focus:ring-quantum-500 border-gray-300 rounded"
                              />
                              <span className="ml-2 text-sm text-gray-900">Wikipedia</span>
                            </label>
                            <label className="flex items-center">
                              <input
                                type="checkbox"
                                {...register('data.universe.source')}
                                value="kaggle"
                                className="h-4 w-4 text-quantum-600 focus:ring-quantum-500 border-gray-300 rounded"
                              />
                              <span className="ml-2 text-sm text-gray-900">Kaggle</span>
                            </label>
                          </div>
                        </div>
                        <div>
                          <label className="form-label">Custom Tickers (optional)</label>
                          <input
                            type="text"
                            {...register('data.universe.tickers')}
                            className="form-input"
                            placeholder="AAPL,MSFT,GOOGL"
                          />
                        </div>
                      </div>
                    </div>

                    <div>
                      <h4 className="text-md font-medium text-gray-900 mb-4">Time Range</h4>
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <label className="form-label">Start Date</label>
                          <input
                            type="date"
                            {...register('data.history.start')}
                            className="form-input"
                          />
                        </div>
                        <div>
                          <label className="form-label">End Date</label>
                          <input
                            type="date"
                            {...register('data.history.end')}
                            className="form-input"
                          />
                        </div>
                      </div>
                    </div>

                    <div>
                      <h4 className="text-md font-medium text-gray-900 mb-4">Risk-Free Rate</h4>
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <label className="form-label">Source</label>
                          <select {...register('data.risk_free.source')} className="form-input">
                            <option value="fred">FRED</option>
                            <option value="yahoo">Yahoo Finance</option>
                          </select>
                        </div>
                        <div>
                          <label className="form-label">Series</label>
                          <input
                            type="text"
                            {...register('data.risk_free.series')}
                            className="form-input"
                            placeholder="DTB3"
                          />
                        </div>
                      </div>
                    </div>

                    <div>
                      <h4 className="text-md font-medium text-gray-900 mb-4">Caching</h4>
                      <div className="space-y-4">
                        <div className="flex items-center">
                          <input
                            type="checkbox"
                            {...register('data.cache.enable')}
                            className="h-4 w-4 text-quantum-600 focus:ring-quantum-500 border-gray-300 rounded"
                          />
                          <label className="ml-2 block text-sm text-gray-900">
                            Enable caching
                          </label>
                        </div>
                        <div>
                          <label className="form-label">Cache Path</label>
                          <input
                            type="text"
                            {...register('data.cache.path')}
                            className="form-input"
                            placeholder="data/interim"
                          />
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {activeTab === 'optimizer' && (
                  <div className="space-y-6">
                    <div>
                      <h4 className="text-md font-medium text-gray-900 mb-4">Objective Function</h4>
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <label className="form-label">Risk Aversion (λ)</label>
                          <input
                            type="range"
                            min="0"
                            max="10"
                            step="0.1"
                            {...register('optimizer.objective.risk_aversion', { valueAsNumber: true })}
                            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                          />
                          <div className="flex justify-between text-xs text-gray-500 mt-1">
                            <span>Conservative (0)</span>
                            <span className="font-medium">{watch('optimizer.objective.risk_aversion')}</span>
                            <span>Aggressive (10)</span>
                          </div>
                        </div>
                        <div>
                          <label className="form-label">Transaction Costs (bps)</label>
                          <input
                            type="number"
                            min="0"
                            max="100"
                            step="0.1"
                            {...register('optimizer.objective.transaction_cost_bps', { valueAsNumber: true })}
                            className="form-input"
                          />
                        </div>
                      </div>
                    </div>

                    <div>
                      <h4 className="text-md font-medium text-gray-900 mb-4">Constraints</h4>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                          <label className="form-label">Portfolio Size (K)</label>
                          <input
                            type="number"
                            min="1"
                            max="100"
                            {...register('optimizer.constraints.cardinality_k', { valueAsNumber: true })}
                            className="form-input"
                          />
                        </div>
                        <div>
                          <label className="form-label">Weight Bounds</label>
                          <div className="grid grid-cols-2 gap-2">
                            <input
                              type="number"
                              min="0"
                              max="1"
                              step="0.01"
                              {...register('optimizer.constraints.weight_bounds.0', { valueAsNumber: true })}
                              className="form-input"
                              placeholder="Min"
                            />
                            <input
                              type="number"
                              min="0"
                              max="1"
                              step="0.01"
                              {...register('optimizer.constraints.weight_bounds.1', { valueAsNumber: true })}
                              className="form-input"
                              placeholder="Max"
                            />
                          </div>
                        </div>
                      </div>
                      <div className="mt-4">
                        <div className="flex items-center">
                          <input
                            type="checkbox"
                            {...register('optimizer.constraints.no_short')}
                            className="h-4 w-4 text-quantum-600 focus:ring-quantum-500 border-gray-300 rounded"
                          />
                          <label className="ml-2 block text-sm text-gray-900">
                            No short selling
                          </label>
                        </div>
                      </div>
                    </div>

                    <div>
                      <h4 className="text-md font-medium text-gray-900 mb-4">QAOA Solver</h4>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                          <label className="form-label">Backend</label>
                          <select {...register('optimizer.solver.backend')} className="form-input">
                            <option value="aer_qasm">Aer QASM Simulator</option>
                            <option value="aer_statevector">Aer Statevector</option>
                            <option value="ibmq_qasm_simulator">IBM QASM Simulator</option>
                          </select>
                        </div>
                        <div>
                          <label className="form-label">Circuit Depth (p)</label>
                          <input
                            type="number"
                            min="1"
                            max="10"
                            {...register('optimizer.solver.p_depth', { valueAsNumber: true })}
                            className="form-input"
                          />
                        </div>
                        <div>
                          <label className="form-label">Number of Shots</label>
                          <select {...register('optimizer.solver.shots', { valueAsNumber: true })} className="form-input">
                            <option value={1024}>1,024</option>
                            <option value={2048}>2,048</option>
                            <option value={4096}>4,096</option>
                            <option value={8192}>8,192</option>
                          </select>
                        </div>
                        <div>
                          <label className="form-label">Optimizer</label>
                          <select {...register('optimizer.solver.optimizer')} className="form-input">
                            <option value="COBYLA">COBYLA</option>
                            <option value="SPSA">SPSA</option>
                            <option value="L_BFGS_B">L-BFGS-B</option>
                          </select>
                        </div>
                      </div>
                    </div>

                    <div>
                      <h4 className="text-md font-medium text-gray-900 mb-4">Hardware</h4>
                      <div className="space-y-4">
                        <div className="flex items-center">
                          <input
                            type="checkbox"
                            {...register('optimizer.hardware.use_ibm')}
                            className="h-4 w-4 text-quantum-600 focus:ring-quantum-500 border-gray-300 rounded"
                          />
                          <label className="ml-2 block text-sm text-gray-900">
                            Use IBM Quantum Hardware
                          </label>
                        </div>
                        <div>
                          <label className="form-label">Backend Name</label>
                          <input
                            type="text"
                            {...register('optimizer.hardware.backend_name')}
                            className="form-input"
                            placeholder="ibmq_lima"
                          />
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {activeTab === 'backtest' && (
                  <div className="space-y-6">
                    <div>
                      <h4 className="text-md font-medium text-gray-900 mb-4">Backtest Settings</h4>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                          <label className="form-label">Rebalancing Frequency</label>
                          <select {...register('backtest.rebalance')} className="form-input">
                            <option value="daily">Daily</option>
                            <option value="weekly">Weekly</option>
                            <option value="monthly">Monthly</option>
                            <option value="quarterly">Quarterly</option>
                          </select>
                        </div>
                        <div>
                          <label className="form-label">Transaction Costs (bps)</label>
                          <input
                            type="number"
                            min="0"
                            max="100"
                            step="0.1"
                            {...register('backtest.costs_bps', { valueAsNumber: true })}
                            className="form-input"
                          />
                        </div>
                      </div>
                    </div>

                    <div>
                      <h4 className="text-md font-medium text-gray-900 mb-4">Time Windows</h4>
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <label className="form-label">Training Period (months)</label>
                          <input
                            type="number"
                            min="1"
                            max="60"
                            {...register('backtest.window.train_months', { valueAsNumber: true })}
                            className="form-input"
                          />
                        </div>
                        <div>
                          <label className="form-label">Testing Period (months)</label>
                          <input
                            type="number"
                            min="1"
                            max="24"
                            {...register('backtest.window.test_months', { valueAsNumber: true })}
                            className="form-input"
                          />
                        </div>
                      </div>
                    </div>

                    <div>
                      <h4 className="text-md font-medium text-gray-900 mb-4">Analysis Options</h4>
                      <div className="space-y-4">
                        <div className="flex items-center">
                          <input
                            type="checkbox"
                            {...register('backtest.rolling')}
                            className="h-4 w-4 text-quantum-600 focus:ring-quantum-500 border-gray-300 rounded"
                          />
                          <label className="ml-2 block text-sm text-gray-900">
                            Use rolling window (walk-forward analysis)
                          </label>
                        </div>
                      </div>
                    </div>

                    <div>
                      <h4 className="text-md font-medium text-gray-900 mb-4">Benchmarks & Metrics</h4>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                          <label className="form-label">Benchmark Strategies</label>
                          <div className="space-y-2">
                            <label className="flex items-center">
                              <input
                                type="checkbox"
                                {...register('backtest.benchmarks')}
                                value="mvo"
                                className="h-4 w-4 text-quantum-600 focus:ring-quantum-500 border-gray-300 rounded"
                              />
                              <span className="ml-2 text-sm text-gray-900">MVO</span>
                            </label>
                            <label className="flex items-center">
                              <input
                                type="checkbox"
                                {...register('backtest.benchmarks')}
                                value="greedy_k"
                                className="h-4 w-4 text-quantum-600 focus:ring-quantum-500 border-gray-300 rounded"
                              />
                              <span className="ml-2 text-sm text-gray-900">Greedy K-Selection</span>
                            </label>
                          </div>
                        </div>
                        <div>
                          <label className="form-label">Performance Metrics</label>
                          <div className="space-y-2">
                            <label className="flex items-center">
                              <input
                                type="checkbox"
                                {...register('backtest.metrics')}
                                value="sharpe"
                                className="h-4 w-4 text-quantum-600 focus:ring-quantum-500 border-gray-300 rounded"
                              />
                              <span className="ml-2 text-sm text-gray-900">Sharpe Ratio</span>
                            </label>
                            <label className="flex items-center">
                              <input
                                type="checkbox"
                                {...register('backtest.metrics')}
                                value="sortino"
                                className="h-4 w-4 text-quantum-600 focus:ring-quantum-500 border-gray-300 rounded"
                              />
                              <span className="ml-2 text-sm text-gray-900">Sortino Ratio</span>
                            </label>
                            <label className="flex items-center">
                              <input
                                type="checkbox"
                                {...register('backtest.metrics')}
                                value="max_drawdown"
                                className="h-4 w-4 text-quantum-600 focus:ring-quantum-500 border-gray-300 rounded"
                              />
                              <span className="ml-2 text-sm text-gray-900">Max Drawdown</span>
                            </label>
                            <label className="flex items-center">
                              <input
                                type="checkbox"
                                {...register('backtest.metrics')}
                                value="turnover"
                                className="h-4 w-4 text-quantum-600 focus:ring-quantum-500 border-gray-300 rounded"
                              />
                              <span className="ml-2 text-sm text-gray-900">Turnover</span>
                            </label>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </form>
            </div>
          </div>
        </div>
      </div>

      {/* Configuration Help */}
      <div className="card">
        <div className="card-header">
          <h3 className="text-lg font-medium text-gray-900">Configuration Help</h3>
        </div>
        <div className="card-body">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div>
              <div className="flex items-center mb-2">
                <InformationCircleIcon className="h-5 w-5 text-blue-500 mr-2" />
                <h4 className="font-medium text-gray-900">Data Configuration</h4>
              </div>
              <p className="text-sm text-gray-600">
                Configure data sources, time ranges, and caching options for market data ingestion.
              </p>
            </div>
            <div>
              <div className="flex items-center mb-2">
                <InformationCircleIcon className="h-5 w-5 text-quantum-500 mr-2" />
                <h4 className="font-medium text-gray-900">Optimizer Configuration</h4>
              </div>
              <p className="text-sm text-gray-600">
                Set up quantum and classical optimization parameters, constraints, and solver settings.
              </p>
            </div>
            <div>
              <div className="flex items-center mb-2">
                <InformationCircleIcon className="h-5 w-5 text-green-500 mr-2" />
                <h4 className="font-medium text-gray-900">Backtest Configuration</h4>
              </div>
              <p className="text-sm text-gray-600">
                Define backtesting parameters, time windows, benchmarks, and performance metrics.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
