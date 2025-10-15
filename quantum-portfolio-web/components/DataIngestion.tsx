'use client'

import { useState } from 'react'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'
import toast from 'react-hot-toast'
import { 
  CloudArrowDownIcon, 
  ChartBarIcon, 
  ClockIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon
} from '@heroicons/react/24/outline'
import { QuantumLoadingScreen, useQuantumLoading } from './QuantumLoadingScreen'

const dataIngestionSchema = z.object({
  startDate: z.string().min(1, 'Start date is required'),
  endDate: z.string().min(1, 'End date is required'),
  dataSource: z.enum(['wikipedia', 'kaggle', 'both']),
  cacheEnabled: z.boolean(),
  cachePath: z.string().optional(),
})

type DataIngestionForm = z.infer<typeof dataIngestionSchema>

interface DataIngestionJob {
  id: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  startTime: string
  endTime?: string
  progress: number
  message: string
}

export function DataIngestion() {
  const [isLoading, setIsLoading] = useState(false)
  const [currentJob, setCurrentJob] = useState<DataIngestionJob | null>(null)
  
  // Quantum loading hook
  const quantumLoading = useQuantumLoading()
  
  const {
    register,
    handleSubmit,
    formState: { errors },
    watch,
  } = useForm<DataIngestionForm>({
    resolver: zodResolver(dataIngestionSchema),
    defaultValues: {
      startDate: '2018-01-01',
      endDate: '2025-01-01',
      dataSource: 'both',
      cacheEnabled: true,
      cachePath: 'data/interim',
    },
  })

  const cacheEnabled = watch('cacheEnabled')

  const onSubmit = async (data: DataIngestionForm) => {
    setIsLoading(true)
    
    // Start quantum loading for data ingestion
    quantumLoading.startLoading('data-ingestion', 'Starting data ingestion...')
    
    const job: DataIngestionJob = {
      id: `data-${Date.now()}`,
      status: 'running',
      startTime: new Date().toISOString(),
      progress: 0,
      message: 'Starting data ingestion...',
    }
    
    setCurrentJob(job)
    
    try {
      // Simulate API call to QEPO backend
      const response = await fetch('/api/qepo/data', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          history: {
            start: data.startDate,
            end: data.endDate,
          },
          universe: {
            source: data.dataSource === 'both' ? ['wikipedia', 'kaggle'] : [data.dataSource],
          },
          cache: {
            enable: data.cacheEnabled,
            path: data.cachePath,
          },
        }),
      })

      if (!response.ok) {
        throw new Error('Failed to start data ingestion')
      }

      const result = await response.json()
      
      setCurrentJob({
        ...job,
        status: 'completed',
        endTime: new Date().toISOString(),
        progress: 100,
        message: `Data ingestion completed successfully. Downloaded ${result.numPriceRecords} price records for ${result.numTickers} tickers.`,
      })
      
      quantumLoading.updateProgress(100, `Data ingestion completed successfully. Downloaded ${result.numPriceRecords} price records for ${result.numTickers} tickers.`)
      setTimeout(() => quantumLoading.stopLoading(), 1000)
      toast.success('Data ingestion completed successfully!')
    } catch (error) {
      setCurrentJob({
        ...job,
        status: 'failed',
        endTime: new Date().toISOString(),
        message: `Data ingestion failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
      })
      
      quantumLoading.stopLoading()
      toast.error('Data ingestion failed')
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
        <CloudArrowDownIcon className="mx-auto h-12 w-12 text-quantum-500" />
        <h2 className="mt-2 text-3xl font-bold text-gray-900">Data Ingestion</h2>
        <p className="mt-2 text-sm text-gray-600">
          Download and prepare market data for portfolio optimization
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Configuration Form */}
        <div className="card">
          <div className="card-header">
            <h3 className="text-lg font-medium text-gray-900">Configuration</h3>
          </div>
          <div className="card-body">
            <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="form-label">Start Date</label>
                  <input
                    type="date"
                    {...register('startDate')}
                    className="form-input"
                  />
                  {errors.startDate && (
                    <p className="mt-1 text-sm text-red-600">{errors.startDate.message}</p>
                  )}
                </div>
                <div>
                  <label className="form-label">End Date</label>
                  <input
                    type="date"
                    {...register('endDate')}
                    className="form-input"
                  />
                  {errors.endDate && (
                    <p className="mt-1 text-sm text-red-600">{errors.endDate.message}</p>
                  )}
                </div>
              </div>

              <div>
                <label className="form-label">Data Source</label>
                <select {...register('dataSource')} className="form-input">
                  <option value="wikipedia">Wikipedia (S&P 500)</option>
                  <option value="kaggle">Kaggle CSV</option>
                  <option value="both">Both Sources</option>
                </select>
              </div>

              <div className="flex items-center">
                <input
                  type="checkbox"
                  {...register('cacheEnabled')}
                  className="h-4 w-4 text-quantum-600 focus:ring-quantum-500 border-gray-300 rounded"
                />
                <label className="ml-2 block text-sm text-gray-900">
                  Enable caching
                </label>
              </div>

              {cacheEnabled && (
                <div>
                  <label className="form-label">Cache Path</label>
                  <input
                    type="text"
                    {...register('cachePath')}
                    className="form-input"
                    placeholder="data/interim"
                  />
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
                    Starting Data Ingestion...
                  </>
                ) : (
                  <>
                    <CloudArrowDownIcon className="h-4 w-4 mr-2" />
                    Start Data Ingestion
                  </>
                )}
              </button>
            </form>
          </div>
        </div>

        {/* Job Status */}
        <div className="card">
          <div className="card-header">
            <h3 className="text-lg font-medium text-gray-900">Job Status</h3>
          </div>
          <div className="card-body">
            {currentJob ? (
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

                <div className="text-sm text-gray-600">
                  <p><strong>Started:</strong> {new Date(currentJob.startTime).toLocaleString()}</p>
                  {currentJob.endTime && (
                    <p><strong>Completed:</strong> {new Date(currentJob.endTime).toLocaleString()}</p>
                  )}
                </div>

                <div className="bg-gray-50 p-3 rounded-md">
                  <p className="text-sm text-gray-700">{currentJob.message}</p>
                </div>
              </div>
            ) : (
              <div className="text-center text-gray-500">
                <ChartBarIcon className="mx-auto h-12 w-12 text-gray-400" />
                <p className="mt-2">No active jobs</p>
                <p className="text-sm">Start a data ingestion job to see status here</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Data Sources Info */}
      <div className="card">
        <div className="card-header">
          <h3 className="text-lg font-medium text-gray-900">Data Sources</h3>
        </div>
        <div className="card-body">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="text-center">
              <div className="text-2xl mb-2">📊</div>
              <h4 className="font-medium text-gray-900">Wikipedia</h4>
              <p className="text-sm text-gray-600">S&P 500 constituent list</p>
            </div>
            <div className="text-center">
              <div className="text-2xl mb-2">📈</div>
              <h4 className="font-medium text-gray-900">yfinance</h4>
              <p className="text-sm text-gray-600">Historical price data</p>
            </div>
            <div className="text-center">
              <div className="text-2xl mb-2">🏛️</div>
              <h4 className="font-medium text-gray-900">FRED</h4>
              <p className="text-sm text-gray-600">Risk-free rate data</p>
            </div>
          </div>
        </div>
      </div>
      
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
