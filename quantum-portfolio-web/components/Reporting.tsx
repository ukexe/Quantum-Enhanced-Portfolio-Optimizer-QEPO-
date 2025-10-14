'use client'

import { useState, useEffect } from 'react'
import { useQuery } from 'react-query'
import toast from 'react-hot-toast'
import { 
  DocumentTextIcon, 
  ChartBarIcon,
  ArrowDownTrayIcon,
  EyeIcon,
  CalendarIcon,
  ClockIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon
} from '@heroicons/react/24/outline'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell } from 'recharts'

interface MLflowRun {
  run_id: string
  start_time: string
  end_time?: string
  status: string
  command: string
  metrics: Record<string, number>
  params: Record<string, string>
  tags: Record<string, string>
  artifacts: string[]
}

interface ReportData {
  run_id: string
  title: string
  summary: {
    total_return: number
    annualized_return: number
    volatility: number
    sharpe_ratio: number
    max_drawdown: number
    num_assets: number
  }
  charts: {
    equity_curve: Array<{ date: string; value: number }>
    monthly_returns: Array<{ month: string; return: number }>
    asset_allocation: Array<{ asset: string; weight: number }>
    risk_metrics: Array<{ metric: string; value: number }>
  }
  details: {
    strategy: string
    parameters: Record<string, any>
    constraints: Record<string, any>
    performance_attribution: Record<string, number>
  }
}

const COLORS = ['#0ea5e9', '#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981', '#ef4444', '#6b7280']

export function Reporting() {
  const [selectedRunId, setSelectedRunId] = useState<string>('')
  const [reportFormat, setReportFormat] = useState<'html' | 'markdown' | 'pdf'>('html')
  const [includeCharts, setIncludeCharts] = useState(true)

  // Fetch MLflow runs
  const { data: runs, isLoading: runsLoading, refetch: refetchRuns } = useQuery(
    'mlflow-runs',
    async () => {
      const response = await fetch('/api/qepo/mlflow/runs')
      if (!response.ok) throw new Error('Failed to fetch runs')
      return response.json() as Promise<MLflowRun[]>
    },
    {
      refetchInterval: 30000, // Refetch every 30 seconds
    }
  )

  // Fetch report data for selected run
  const { data: reportData, isLoading: reportLoading } = useQuery(
    ['report-data', selectedRunId],
    async () => {
      if (!selectedRunId) return null
      const response = await fetch(`/api/qepo/report/${selectedRunId}`)
      if (!response.ok) throw new Error('Failed to fetch report data')
      return response.json() as Promise<ReportData>
    },
    {
      enabled: !!selectedRunId,
    }
  )

  const generateReport = async () => {
    if (!selectedRunId) {
      toast.error('Please select a run to generate a report for')
      return
    }

    try {
      const response = await fetch('/api/qepo/report/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          run_id: selectedRunId,
          format: reportFormat,
          include_charts: includeCharts,
        }),
      })

      if (!response.ok) {
        throw new Error('Failed to generate report')
      }

      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `qepo-report-${selectedRunId}.${reportFormat === 'html' ? 'html' : reportFormat === 'pdf' ? 'pdf' : 'md'}`
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(url)
      document.body.removeChild(a)

      toast.success('Report generated and downloaded successfully!')
    } catch (error) {
      toast.error('Failed to generate report')
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status.toLowerCase()) {
      case 'finished':
        return <CheckCircleIcon className="h-4 w-4 text-green-500" />
      case 'failed':
        return <ExclamationTriangleIcon className="h-4 w-4 text-red-500" />
      case 'running':
        return <div className="quantum-spinner" />
      default:
        return <ClockIcon className="h-4 w-4 text-gray-400" />
    }
  }

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString()
  }

  const formatDuration = (startTime: string, endTime?: string) => {
    const start = new Date(startTime)
    const end = endTime ? new Date(endTime) : new Date()
    const duration = end.getTime() - start.getTime()
    const minutes = Math.floor(duration / 60000)
    const seconds = Math.floor((duration % 60000) / 1000)
    return `${minutes}m ${seconds}s`
  }

  return (
    <div className="space-y-6">
      <div className="text-center">
        <DocumentTextIcon className="mx-auto h-12 w-12 text-quantum-500" />
        <h2 className="mt-2 text-3xl font-bold text-gray-900">Reporting & Analytics</h2>
        <p className="mt-2 text-sm text-gray-600">
          Generate comprehensive reports and analyze experiment results
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Run Selection */}
        <div className="lg:col-span-1">
          <div className="card">
            <div className="card-header">
              <h3 className="text-lg font-medium text-gray-900">Experiment Runs</h3>
            </div>
            <div className="card-body">
              <div className="space-y-4">
                <button
                  onClick={() => refetchRuns()}
                  className="btn-secondary w-full"
                >
                  <ChartBarIcon className="h-4 w-4 mr-2" />
                  Refresh Runs
                </button>

                {runsLoading ? (
                  <div className="text-center py-4">
                    <div className="quantum-spinner mx-auto" />
                    <p className="text-sm text-gray-500 mt-2">Loading runs...</p>
                  </div>
                ) : runs && runs.length > 0 ? (
                  <div className="space-y-2 max-h-96 overflow-y-auto">
                    {runs.map((run) => (
                      <div
                        key={run.run_id}
                        onClick={() => setSelectedRunId(run.run_id)}
                        className={`p-3 border rounded-lg cursor-pointer transition-colors ${
                          selectedRunId === run.run_id
                            ? 'border-quantum-500 bg-quantum-50'
                            : 'border-gray-200 hover:border-gray-300'
                        }`}
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center">
                            {getStatusIcon(run.status)}
                            <span className="ml-2 text-sm font-medium text-gray-900">
                              {run.run_id.slice(0, 8)}...
                            </span>
                          </div>
                          <span className={`status-${run.status.toLowerCase()}`}>
                            {run.status}
                          </span>
                        </div>
                        <div className="mt-1 text-xs text-gray-500">
                          <div className="flex items-center">
                            <CalendarIcon className="h-3 w-3 mr-1" />
                            {formatDate(run.start_time)}
                          </div>
                          <div className="flex items-center">
                            <ClockIcon className="h-3 w-3 mr-1" />
                            {formatDuration(run.start_time, run.end_time)}
                          </div>
                          <div className="text-quantum-600 font-medium">
                            {run.command}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-4">
                    <DocumentTextIcon className="mx-auto h-8 w-8 text-gray-400" />
                    <p className="text-sm text-gray-500 mt-2">No runs found</p>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Report Generation */}
          <div className="card mt-6">
            <div className="card-header">
              <h3 className="text-lg font-medium text-gray-900">Generate Report</h3>
            </div>
            <div className="card-body">
              <div className="space-y-4">
                <div>
                  <label className="form-label">Format</label>
                  <select
                    value={reportFormat}
                    onChange={(e) => setReportFormat(e.target.value as 'html' | 'markdown' | 'pdf')}
                    className="form-input"
                  >
                    <option value="html">HTML</option>
                    <option value="markdown">Markdown</option>
                    <option value="pdf">PDF</option>
                  </select>
                </div>

                <div className="flex items-center">
                  <input
                    type="checkbox"
                    checked={includeCharts}
                    onChange={(e) => setIncludeCharts(e.target.checked)}
                    className="h-4 w-4 text-quantum-600 focus:ring-quantum-500 border-gray-300 rounded"
                  />
                  <label className="ml-2 block text-sm text-gray-900">
                    Include charts
                  </label>
                </div>

                <button
                  onClick={generateReport}
                  disabled={!selectedRunId}
                  className="btn-quantum w-full"
                >
                  <ArrowDownTrayIcon className="h-4 w-4 mr-2" />
                  Generate Report
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Report Preview */}
        <div className="lg:col-span-2">
          {selectedRunId && reportData ? (
            <div className="space-y-6">
              {/* Summary Metrics */}
              <div className="card">
                <div className="card-header">
                  <h3 className="text-lg font-medium text-gray-900">Performance Summary</h3>
                </div>
                <div className="card-body">
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-6">
                    <div className="metric-card text-center">
                      <div className="metric-value text-emerald-600">
                        {((reportData.summary.total_return || 0) * 100).toFixed(1)}%
                      </div>
                      <div className="metric-label">Total Return</div>
                    </div>
                    <div className="metric-card text-center">
                      <div className="metric-value text-blue-600">
                        {((reportData.summary.annualized_return || 0) * 100).toFixed(1)}%
                      </div>
                      <div className="metric-label">Annualized Return</div>
                    </div>
                    <div className="metric-card text-center">
                      <div className="metric-value text-purple-600">
                        {(reportData.summary.sharpe_ratio || 0).toFixed(2)}
                      </div>
                      <div className="metric-label">Sharpe Ratio</div>
                    </div>
                    <div className="metric-card text-center">
                      <div className="metric-value text-amber-600">
                        {((reportData.summary.volatility || 0) * 100).toFixed(1)}%
                      </div>
                      <div className="metric-label">Volatility</div>
                    </div>
                    <div className="metric-card text-center">
                      <div className="metric-value text-red-600">
                        {((reportData.summary.max_drawdown || 0) * 100).toFixed(1)}%
                      </div>
                      <div className="metric-label">Max Drawdown</div>
                    </div>
                    <div className="metric-card text-center">
                      <div className="metric-value text-indigo-600">
                        {reportData.summary.num_assets || 0}
                      </div>
                      <div className="metric-label">Assets</div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Equity Curve */}
              <div className="card">
                <div className="card-header">
                  <h3 className="text-lg font-medium text-gray-900">Equity Curve</h3>
                </div>
                <div className="card-body">
                  <div className="h-80">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={reportData.charts.equity_curve}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="date" />
                        <YAxis />
                        <Tooltip 
                          formatter={(value: number) => [`${(value * 100).toFixed(2)}%`, '']}
                          labelFormatter={(label) => `Date: ${label}`}
                        />
                        <Line 
                          type="monotone" 
                          dataKey="value" 
                          stroke="#0ea5e9" 
                          strokeWidth={2}
                          dot={false}
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
                      <BarChart data={reportData.charts.monthly_returns}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="month" />
                        <YAxis />
                        <Tooltip 
                          formatter={(value: number) => [`${(value * 100).toFixed(2)}%`, '']}
                        />
                        <Bar 
                          dataKey="return" 
                          fill="#0ea5e9"
                          name="Monthly Return"
                        />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>

              {/* Asset Allocation */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="card">
                  <div className="card-header">
                    <h3 className="text-lg font-medium text-gray-900">Asset Allocation</h3>
                  </div>
                  <div className="card-body">
                    <div className="h-64">
                      <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                          <Pie
                            data={reportData.charts.asset_allocation}
                            cx="50%"
                            cy="50%"
                            labelLine={false}
                            label={({ asset, percent }) => `${asset} ${(percent * 100).toFixed(0)}%`}
                            outerRadius={80}
                            fill="#8884d8"
                            dataKey="weight"
                          >
                            {reportData.charts.asset_allocation.map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                            ))}
                          </Pie>
                          <Tooltip formatter={(value: number) => [`${(value * 100).toFixed(2)}%`, 'Weight']} />
                        </PieChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </div>

                <div className="card">
                  <div className="card-header">
                    <h3 className="text-lg font-medium text-gray-900">Risk Metrics</h3>
                  </div>
                  <div className="card-body">
                    <div className="h-64">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={reportData.charts.risk_metrics} layout="horizontal">
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis type="number" />
                          <YAxis dataKey="metric" type="category" width={100} />
                          <Tooltip 
                            formatter={(value: number) => [value.toFixed(3), '']}
                          />
                          <Bar dataKey="value" fill="#0ea5e9" />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </div>
              </div>

              {/* Strategy Details */}
              <div className="card">
                <div className="card-header">
                  <h3 className="text-lg font-medium text-gray-900">Strategy Details</h3>
                </div>
                <div className="card-body">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <h4 className="font-medium text-gray-900 mb-2">Parameters</h4>
                      <div className="space-y-1 text-sm">
                        {Object.entries(reportData.details.parameters).map(([key, value]) => (
                          <div key={key} className="flex justify-between">
                            <span className="text-gray-600">{key}:</span>
                            <span className="font-medium">{String(value)}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                    <div>
                      <h4 className="font-medium text-gray-900 mb-2">Constraints</h4>
                      <div className="space-y-1 text-sm">
                        {Object.entries(reportData.details.constraints).map(([key, value]) => (
                          <div key={key} className="flex justify-between">
                            <span className="text-gray-600">{key}:</span>
                            <span className="font-medium">{String(value)}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ) : selectedRunId ? (
            <div className="card">
              <div className="card-body text-center py-12">
                <div className="quantum-spinner mx-auto" />
                <p className="text-sm text-gray-500 mt-2">Loading report data...</p>
              </div>
            </div>
          ) : (
            <div className="card">
              <div className="card-body text-center py-12">
                <EyeIcon className="mx-auto h-12 w-12 text-gray-400" />
                <p className="text-sm text-gray-500 mt-2">Select a run to view report</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
