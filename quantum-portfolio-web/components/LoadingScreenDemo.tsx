'use client'

import { useState } from 'react'
import { QuantumLoadingScreen, useQuantumLoading } from './QuantumLoadingScreen'
import { 
  CpuChipIcon, 
  BeakerIcon, 
  ChartBarIcon,
  BoltIcon,
  SparklesIcon,
  CubeIcon,
  PlayIcon
} from '@heroicons/react/24/outline'

const algorithms = [
  { id: 'qaoa', name: 'QAOA Optimization', icon: CpuChipIcon, color: 'from-cyan-500 to-blue-600' },
  { id: 'mvo', name: 'Modern Portfolio Theory', icon: ChartBarIcon, color: 'from-emerald-500 to-teal-600' },
  { id: 'greedy', name: 'Greedy Algorithm', icon: BoltIcon, color: 'from-amber-500 to-orange-600' },
  { id: 'data-ingestion', name: 'Data Processing', icon: BeakerIcon, color: 'from-purple-500 to-indigo-600' },
  { id: 'backtest', name: 'Backtesting', icon: CubeIcon, color: 'from-rose-500 to-pink-600' },
  { id: 'general', name: 'General Processing', icon: SparklesIcon, color: 'from-slate-500 to-gray-600' },
]

export function LoadingScreenDemo() {
  const quantumLoading = useQuantumLoading()
  const [isRunning, setIsRunning] = useState(false)

  const startDemo = async (algorithm: typeof algorithms[0]) => {
    setIsRunning(true)
    quantumLoading.startLoading(algorithm.id as any, `Starting ${algorithm.name}...`)
    
    // Simulate different processing times
    const durations = {
      'qaoa': 8000,
      'mvo': 5000,
      'greedy': 3000,
      'data-ingestion': 6000,
      'backtest': 7000,
      'general': 4000
    }
    
    const duration = durations[algorithm.id as keyof typeof durations] || 4000
    const steps = [
      'Initializing...',
      'Loading data...',
      'Processing...',
      'Optimizing...',
      'Validating results...',
      'Finalizing...'
    ]
    
    for (let i = 0; i < steps.length; i++) {
      const progress = Math.round(((i + 1) / steps.length) * 100)
      quantumLoading.updateProgress(progress, steps[i])
      await new Promise(resolve => setTimeout(resolve, duration / steps.length))
    }
    
    quantumLoading.updateProgress(100, `${algorithm.name} completed successfully!`)
    setTimeout(() => {
      quantumLoading.stopLoading()
      setIsRunning(false)
    }, 1000)
  }

  return (
    <div className="space-y-8">
      <div className="text-center">
        <h2 className="text-3xl font-bold text-gray-900 mb-4">
          Quantum Loading Screen Demo
        </h2>
        <p className="text-gray-600 max-w-2xl mx-auto">
          Experience the modern quantum-themed loading screen across different algorithm types. 
          Each demo simulates realistic processing times and progress updates.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {algorithms.map((algorithm) => {
          const IconComponent = algorithm.icon
          return (
            <div
              key={algorithm.id}
              className="card hover:shadow-lg transition-all duration-200 cursor-pointer group"
              onClick={() => !isRunning && startDemo(algorithm)}
            >
              <div className="card-body text-center">
                <div className={`inline-flex p-4 rounded-2xl bg-gradient-to-r ${algorithm.color} mb-4 group-hover:scale-105 transition-transform duration-200`}>
                  <IconComponent className="w-8 h-8 text-white" />
                </div>
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  {algorithm.name}
                </h3>
                <p className="text-sm text-gray-600 mb-4">
                  Click to start a demo of this algorithm's loading screen
                </p>
                <button
                  disabled={isRunning}
                  className={`btn btn-primary w-full ${isRunning ? 'opacity-50 cursor-not-allowed' : ''}`}
                >
                  <PlayIcon className="w-4 h-4 mr-2" />
                  {isRunning ? 'Running...' : 'Start Demo'}
                </button>
              </div>
            </div>
          )
        })}
      </div>

      <div className="card">
        <div className="card-header">
          <h3 className="text-lg font-medium text-gray-900">Features</h3>
        </div>
        <div className="card-body">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-semibold text-gray-900 mb-3">Visual Effects</h4>
              <ul className="space-y-2 text-sm text-gray-600">
                <li>• Quantum particle animations</li>
                <li>• Algorithm-specific color schemes</li>
                <li>• Glass morphism design</li>
                <li>• Smooth progress transitions</li>
                <li>• Floating background elements</li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold text-gray-900 mb-3">Functionality</h4>
              <ul className="space-y-2 text-sm text-gray-600">
                <li>• Real-time progress updates</li>
                <li>• Dynamic status messages</li>
                <li>• Cancel operation support</li>
                <li>• Algorithm-specific configurations</li>
                <li>• Responsive design</li>
              </ul>
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
          setIsRunning(false)
        }}
      />
    </div>
  )
}
