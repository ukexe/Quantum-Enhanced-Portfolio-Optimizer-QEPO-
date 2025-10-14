'use client'

import { useState, useEffect } from 'react'
import { useQuery } from 'react-query'
import toast from 'react-hot-toast'
import { 
  PlayIcon, 
  ClockIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  XMarkIcon,
  EyeIcon,
  ArrowPathIcon
} from '@heroicons/react/24/outline'

interface Job {
  id: string
  type: 'data' | 'optimize' | 'backtest' | 'report'
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
  startTime: string
  endTime?: string
  progress: number
  message: string
  config: Record<string, any>
  results?: Record<string, any>
  logs: string[]
}

interface JobStats {
  total: number
  running: number
  completed: number
  failed: number
  pending: number
  cancelled: number
}

export function JobMonitor() {
  const [selectedJob, setSelectedJob] = useState<Job | null>(null)
  const [showLogs, setShowLogs] = useState(false)
  const [autoRefresh, setAutoRefresh] = useState(true)

  // Fetch jobs with auto-refresh
  const { data: jobs, isLoading, refetch } = useQuery(
    'jobs',
    async () => {
      const response = await fetch('/api/qepo/jobs')
      if (!response.ok) throw new Error('Failed to fetch jobs')
      return response.json() as Promise<Job[]>
    },
    {
      refetchInterval: autoRefresh ? 2000 : false, // Refresh every 2 seconds if enabled
      refetchOnWindowFocus: true,
    }
  )

  // Calculate job statistics
  const jobStats: JobStats = jobs?.reduce(
    (stats, job) => {
      stats.total++
      stats[job.status as keyof JobStats]++
      return stats
    },
    { total: 0, running: 0, completed: 0, failed: 0, pending: 0, cancelled: 0 }
  ) || { total: 0, running: 0, completed: 0, failed: 0, pending: 0, cancelled: 0 }

  const cancelJob = async (jobId: string) => {
    try {
      const response = await fetch(`/api/qepo/jobs/${jobId}/cancel`, {
        method: 'POST',
      })
      if (!response.ok) throw new Error('Failed to cancel job')
      toast.success('Job cancelled successfully')
      refetch()
    } catch (error) {
      toast.error('Failed to cancel job')
    }
  }

  const deleteJob = async (jobId: string) => {
    try {
      const response = await fetch(`/api/qepo/jobs/${jobId}`, {
        method: 'DELETE',
      })
      if (!response.ok) throw new Error('Failed to delete job')
      toast.success('Job deleted successfully')
      refetch()
    } catch (error) {
      toast.error('Failed to delete job')
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircleIcon className="h-4 w-4 text-green-500" />
      case 'failed':
        return <ExclamationTriangleIcon className="h-4 w-4 text-red-500" />
      case 'running':
        return <div className="quantum-spinner" />
      case 'cancelled':
        return <XMarkIcon className="h-4 w-4 text-gray-500" />
      default:
        return <ClockIcon className="h-4 w-4 text-gray-400" />
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'bg-green-100 text-green-800'
      case 'failed':
        return 'bg-red-100 text-red-800'
      case 'running':
        return 'bg-yellow-100 text-yellow-800'
      case 'cancelled':
        return 'bg-gray-100 text-gray-800'
      default:
        return 'bg-gray-100 text-gray-800'
    }
  }

  const getJobTypeIcon = (type: string) => {
    switch (type) {
      case 'data':
        return '📊'
      case 'optimize':
        return '⚛️'
      case 'backtest':
        return '🧪'
      case 'report':
        return '📄'
      default:
        return '🔧'
    }
  }

  const formatDuration = (startTime: string, endTime?: string) => {
    const start = new Date(startTime)
    const end = endTime ? new Date(endTime) : new Date()
    const duration = end.getTime() - start.getTime()
    const minutes = Math.floor(duration / 60000)
    const seconds = Math.floor((duration % 60000) / 1000)
    return `${minutes}m ${seconds}s`
  }

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString()
  }

  return (
    <div className="space-y-6">
      <div className="text-center">
        <PlayIcon className="mx-auto h-12 w-12 text-quantum-500" />
        <h2 className="mt-2 text-3xl font-bold text-gray-900">Job Monitor</h2>
        <p className="mt-2 text-sm text-gray-600">
          Monitor and manage all QEPO jobs in real-time
        </p>
      </div>

      {/* Job Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
        <div className="card">
          <div className="card-body text-center">
            <div className="text-2xl font-bold text-gray-900">{jobStats.total}</div>
            <div className="text-sm text-gray-600">Total Jobs</div>
          </div>
        </div>
        <div className="card">
          <div className="card-body text-center">
            <div className="text-2xl font-bold text-yellow-600">{jobStats.running}</div>
            <div className="text-sm text-gray-600">Running</div>
          </div>
        </div>
        <div className="card">
          <div className="card-body text-center">
            <div className="text-2xl font-bold text-green-600">{jobStats.completed}</div>
            <div className="text-sm text-gray-600">Completed</div>
          </div>
        </div>
        <div className="card">
          <div className="card-body text-center">
            <div className="text-2xl font-bold text-red-600">{jobStats.failed}</div>
            <div className="text-sm text-gray-600">Failed</div>
          </div>
        </div>
        <div className="card">
          <div className="card-body text-center">
            <div className="text-2xl font-bold text-gray-600">{jobStats.pending}</div>
            <div className="text-sm text-gray-600">Pending</div>
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="flex justify-between items-center">
        <div className="flex items-center space-x-4">
          <button
            onClick={() => refetch()}
            className="btn-secondary"
          >
            <ArrowPathIcon className="h-4 w-4 mr-2" />
            Refresh
          </button>
          <label className="flex items-center">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
              className="h-4 w-4 text-quantum-600 focus:ring-quantum-500 border-gray-300 rounded"
            />
            <span className="ml-2 text-sm text-gray-900">Auto-refresh</span>
          </label>
        </div>
        <div className="text-sm text-gray-500">
          {autoRefresh && 'Auto-refreshing every 2 seconds'}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Job List */}
        <div className="lg:col-span-2">
          <div className="card">
            <div className="card-header">
              <h3 className="text-lg font-medium text-gray-900">Active Jobs</h3>
            </div>
            <div className="card-body">
              {isLoading ? (
                <div className="text-center py-8">
                  <div className="quantum-spinner mx-auto" />
                  <p className="text-sm text-gray-500 mt-2">Loading jobs...</p>
                </div>
              ) : jobs && jobs.length > 0 ? (
                <div className="space-y-3">
                  {jobs.map((job) => (
                    <div
                      key={job.id}
                      className={`p-4 border rounded-lg cursor-pointer transition-colors ${
                        selectedJob?.id === job.id
                          ? 'border-quantum-500 bg-quantum-50'
                          : 'border-gray-200 hover:border-gray-300'
                      }`}
                      onClick={() => setSelectedJob(job)}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-3">
                          <span className="text-lg">{getJobTypeIcon(job.type)}</span>
                          <div>
                            <div className="flex items-center space-x-2">
                              {getStatusIcon(job.status)}
                              <span className="text-sm font-medium text-gray-900">
                                {job.type.toUpperCase()} - {job.id.slice(0, 8)}...
                              </span>
                            </div>
                            <div className="text-xs text-gray-500">
                              Started: {formatDate(job.startTime)}
                            </div>
                          </div>
                        </div>
                        <div className="flex items-center space-x-2">
                          <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(job.status)}`}>
                            {job.status}
                          </span>
                          {job.status === 'running' && (
                            <button
                              onClick={(e) => {
                                e.stopPropagation()
                                cancelJob(job.id)
                              }}
                              className="text-red-600 hover:text-red-700"
                            >
                              <XMarkIcon className="h-4 w-4" />
                            </button>
                          )}
                          {job.status !== 'running' && (
                            <button
                              onClick={(e) => {
                                e.stopPropagation()
                                deleteJob(job.id)
                              }}
                              className="text-gray-400 hover:text-gray-600"
                            >
                              <XMarkIcon className="h-4 w-4" />
                            </button>
                          )}
                        </div>
                      </div>

                      {job.status === 'running' && (
                        <div className="mt-3">
                          <div className="flex justify-between text-sm text-gray-600 mb-1">
                            <span>Progress</span>
                            <span>{job.progress}%</span>
                          </div>
                          <div className="w-full bg-gray-200 rounded-full h-2">
                            <div
                              className="bg-quantum-600 h-2 rounded-full transition-all duration-300"
                              style={{ width: `${job.progress}%` }}
                            />
                          </div>
                        </div>
                      )}

                      <div className="mt-2 text-sm text-gray-600">
                        {job.message}
                      </div>

                      <div className="mt-2 flex justify-between text-xs text-gray-500">
                        <span>Duration: {formatDuration(job.startTime, job.endTime)}</span>
                        <span>Logs: {job.logs.length} entries</span>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8">
                  <PlayIcon className="mx-auto h-12 w-12 text-gray-400" />
                  <p className="text-sm text-gray-500 mt-2">No jobs found</p>
                  <p className="text-xs text-gray-400">Start a job from other tabs to see it here</p>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Job Details */}
        <div className="lg:col-span-1">
          {selectedJob ? (
            <div className="space-y-6">
              {/* Job Details */}
              <div className="card">
                <div className="card-header">
                  <h3 className="text-lg font-medium text-gray-900">Job Details</h3>
                </div>
                <div className="card-body">
                  <div className="space-y-3">
                    <div>
                      <span className="text-sm font-medium text-gray-600">ID:</span>
                      <p className="text-sm text-gray-900 font-mono">{selectedJob.id}</p>
                    </div>
                    <div>
                      <span className="text-sm font-medium text-gray-600">Type:</span>
                      <p className="text-sm text-gray-900">{selectedJob.type.toUpperCase()}</p>
                    </div>
                    <div>
                      <span className="text-sm font-medium text-gray-600">Status:</span>
                      <div className="flex items-center space-x-2">
                        {getStatusIcon(selectedJob.status)}
                        <span className="text-sm text-gray-900">{selectedJob.status}</span>
                      </div>
                    </div>
                    <div>
                      <span className="text-sm font-medium text-gray-600">Started:</span>
                      <p className="text-sm text-gray-900">{formatDate(selectedJob.startTime)}</p>
                    </div>
                    {selectedJob.endTime && (
                      <div>
                        <span className="text-sm font-medium text-gray-600">Ended:</span>
                        <p className="text-sm text-gray-900">{formatDate(selectedJob.endTime)}</p>
                      </div>
                    )}
                    <div>
                      <span className="text-sm font-medium text-gray-600">Duration:</span>
                      <p className="text-sm text-gray-900">
                        {formatDuration(selectedJob.startTime, selectedJob.endTime)}
                      </p>
                    </div>
                    <div>
                      <span className="text-sm font-medium text-gray-600">Message:</span>
                      <p className="text-sm text-gray-900">{selectedJob.message}</p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Configuration */}
              <div className="card">
                <div className="card-header">
                  <h3 className="text-lg font-medium text-gray-900">Configuration</h3>
                </div>
                <div className="card-body">
                  <div className="max-h-64 overflow-y-auto">
                    <pre className="text-xs text-gray-600 bg-gray-50 p-3 rounded">
                      {JSON.stringify(selectedJob.config, null, 2)}
                    </pre>
                  </div>
                </div>
              </div>

              {/* Results */}
              {selectedJob.results && (
                <div className="card">
                  <div className="card-header">
                    <h3 className="text-lg font-medium text-gray-900">Results</h3>
                  </div>
                  <div className="card-body">
                    <div className="max-h-64 overflow-y-auto">
                      <pre className="text-xs text-gray-600 bg-gray-50 p-3 rounded">
                        {JSON.stringify(selectedJob.results, null, 2)}
                      </pre>
                    </div>
                  </div>
                </div>
              )}

              {/* Logs */}
              <div className="card">
                <div className="card-header">
                  <div className="flex items-center justify-between">
                    <h3 className="text-lg font-medium text-gray-900">Logs</h3>
                    <button
                      onClick={() => setShowLogs(!showLogs)}
                      className="text-quantum-600 hover:text-quantum-700"
                    >
                      <EyeIcon className="h-4 w-4" />
                    </button>
                  </div>
                </div>
                {showLogs && (
                  <div className="card-body">
                    <div className="max-h-64 overflow-y-auto bg-gray-900 text-green-400 p-3 rounded font-mono text-xs">
                      {selectedJob.logs.length > 0 ? (
                        selectedJob.logs.map((log, index) => (
                          <div key={index} className="mb-1">
                            {log}
                          </div>
                        ))
                      ) : (
                        <div className="text-gray-500">No logs available</div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="card">
              <div className="card-body text-center py-12">
                <EyeIcon className="mx-auto h-12 w-12 text-gray-400" />
                <p className="text-sm text-gray-500 mt-2">Select a job to view details</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
