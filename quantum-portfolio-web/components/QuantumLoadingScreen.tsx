'use client'

import { useState, useEffect } from 'react'
import { 
  CpuChipIcon, 
  BeakerIcon, 
  ChartBarIcon,
  BoltIcon,
  SparklesIcon,
  CubeIcon
} from '@heroicons/react/24/outline'

interface QuantumLoadingScreenProps {
  isLoading: boolean
  progress?: number
  message?: string
  algorithm?: 'qaoa' | 'mvo' | 'greedy' | 'data-ingestion' | 'backtest' | 'general'
  estimatedTime?: string
  onCancel?: () => void
}

const algorithmConfig = {
  qaoa: {
    title: 'Quantum Annealing Optimization',
    description: 'Running QAOA algorithm on quantum circuits',
    icon: CpuChipIcon,
    color: 'from-cyan-500 to-blue-600',
    particles: 8,
    speed: 'slow'
  },
  mvo: {
    title: 'Modern Portfolio Theory',
    description: 'Optimizing portfolio using mean-variance analysis',
    icon: ChartBarIcon,
    color: 'from-emerald-500 to-teal-600',
    particles: 6,
    speed: 'medium'
  },
  greedy: {
    title: 'Greedy Optimization',
    description: 'Applying greedy selection algorithm',
    icon: BoltIcon,
    color: 'from-amber-500 to-orange-600',
    particles: 4,
    speed: 'fast'
  },
  'data-ingestion': {
    title: 'Data Processing',
    description: 'Ingesting and processing market data',
    icon: BeakerIcon,
    color: 'from-purple-500 to-indigo-600',
    particles: 10,
    speed: 'medium'
  },
  backtest: {
    title: 'Backtesting Analysis',
    description: 'Running historical performance analysis',
    icon: CubeIcon,
    color: 'from-rose-500 to-pink-600',
    particles: 7,
    speed: 'slow'
  },
  general: {
    title: 'Processing',
    description: 'Working on your request',
    icon: SparklesIcon,
    color: 'from-slate-500 to-gray-600',
    particles: 5,
    speed: 'medium'
  }
}

export function QuantumLoadingScreen({ 
  isLoading, 
  progress = 0, 
  message = 'Processing...', 
  algorithm = 'general',
  estimatedTime,
  onCancel 
}: QuantumLoadingScreenProps) {
  const [currentStep, setCurrentStep] = useState(0)
  const [particles, setParticles] = useState<Array<{ id: number; x: number; y: number; delay: number }>>([])
  
  const config = algorithmConfig[algorithm]
  const IconComponent = config.icon

  // Initialize particles
  useEffect(() => {
    const newParticles = Array.from({ length: config.particles }, (_, i) => ({
      id: i,
      x: Math.random() * 100,
      y: Math.random() * 100,
      delay: Math.random() * 2
    }))
    setParticles(newParticles)
  }, [config.particles])

  // Simulate processing steps
  useEffect(() => {
    if (!isLoading) return

    const steps = [
      'Initializing quantum circuits...',
      'Loading market data...',
      'Calculating covariance matrix...',
      'Optimizing portfolio weights...',
      'Validating constraints...',
      'Finalizing results...'
    ]

    const interval = setInterval(() => {
      setCurrentStep(prev => (prev + 1) % steps.length)
    }, 2000)

    return () => clearInterval(interval)
  }, [isLoading])

  if (!isLoading) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/50 backdrop-blur-sm" />
      
      {/* Main Loading Container */}
      <div className="relative z-10 w-full max-w-2xl mx-4">
        <div className="glass-effect rounded-3xl p-8 shadow-2xl border border-white/20">
          {/* Header */}
          <div className="text-center mb-8">
            <div className="relative inline-block mb-4">
              <div className={`p-4 rounded-2xl bg-gradient-to-r ${config.color} shadow-lg`}>
                <IconComponent className="w-12 h-12 text-white" />
              </div>
              <div className="absolute -top-2 -right-2 w-6 h-6 bg-yellow-400 rounded-full animate-pulse" />
            </div>
            
            <h2 className="text-2xl font-bold text-gray-900 mb-2">
              {config.title}
            </h2>
            <p className="text-gray-600">
              {config.description}
            </p>
          </div>

          {/* Progress Section */}
          <div className="space-y-6">
            {/* Progress Bar */}
            <div className="space-y-2">
              <div className="flex justify-between text-sm text-gray-600">
                <span>Progress</span>
                <span>{Math.round(progress)}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
                <div 
                  className={`h-full bg-gradient-to-r ${config.color} rounded-full transition-all duration-500 ease-out relative`}
                  style={{ width: `${progress}%` }}
                >
                  <div className="absolute inset-0 bg-white/30 animate-pulse" />
                </div>
              </div>
            </div>

            {/* Current Message */}
            <div className="text-center">
              <p className="text-lg font-medium text-gray-800 mb-1">
                {message}
              </p>
              {estimatedTime && (
                <p className="text-sm text-gray-500">
                  Estimated time remaining: {estimatedTime}
                </p>
              )}
            </div>

            {/* Quantum Particles Animation */}
            <div className="relative h-32 overflow-hidden rounded-xl bg-gradient-to-br from-gray-50 to-gray-100">
              {particles.map((particle) => (
                <div
                  key={particle.id}
                  className={`absolute w-2 h-2 bg-gradient-to-r ${config.color} rounded-full opacity-60`}
                  style={{
                    left: `${particle.x}%`,
                    top: `${particle.y}%`,
                    animationDelay: `${particle.delay}s`,
                    animation: `quantum-float-${config.speed} 4s ease-in-out infinite`
                  }}
                />
              ))}
              
              {/* Central Quantum Core */}
              <div className="absolute inset-0 flex items-center justify-center">
                <div className={`w-16 h-16 rounded-full bg-gradient-to-r ${config.color} opacity-20 animate-pulse`} />
                <div className={`absolute w-8 h-8 rounded-full bg-gradient-to-r ${config.color} opacity-40 animate-ping`} />
              </div>
            </div>

            {/* Processing Steps */}
            <div className="space-y-2">
              <p className="text-sm font-medium text-gray-700 text-center">
                Current Step:
              </p>
              <div className="flex justify-center">
                <div className="flex space-x-2">
                  {Array.from({ length: 6 }, (_, i) => (
                    <div
                      key={i}
                      className={`w-2 h-2 rounded-full transition-all duration-300 ${
                        i <= currentStep 
                          ? `bg-gradient-to-r ${config.color}` 
                          : 'bg-gray-300'
                      }`}
                    />
                  ))}
                </div>
              </div>
            </div>

            {/* Cancel Button */}
            {onCancel && (
              <div className="text-center pt-4">
                <button
                  onClick={onCancel}
                  className="px-6 py-2 text-sm font-medium text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded-lg transition-colors duration-200"
                >
                  Cancel Operation
                </button>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Floating Elements */}
      <div className="absolute inset-0 pointer-events-none">
        {Array.from({ length: 12 }, (_, i) => (
          <div
            key={i}
            className="absolute w-1 h-1 bg-cyan-400 rounded-full opacity-30"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              animationDelay: `${Math.random() * 3}s`,
              animation: 'quantum-float 8s ease-in-out infinite'
            }}
          />
        ))}
      </div>
    </div>
  )
}

// Hook for managing loading state
export function useQuantumLoading() {
  const [isLoading, setIsLoading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [message, setMessage] = useState('Processing...')
  const [algorithm, setAlgorithm] = useState<QuantumLoadingScreenProps['algorithm']>('general')

  const startLoading = (algo: QuantumLoadingScreenProps['algorithm'] = 'general', initialMessage = 'Processing...') => {
    setAlgorithm(algo)
    setMessage(initialMessage)
    setProgress(0)
    setIsLoading(true)
  }

  const updateProgress = (newProgress: number, newMessage?: string) => {
    setProgress(Math.min(100, Math.max(0, newProgress)))
    if (newMessage) setMessage(newMessage)
  }

  const stopLoading = () => {
    setIsLoading(false)
    setProgress(0)
    setMessage('Processing...')
  }

  return {
    isLoading,
    progress,
    message,
    algorithm,
    startLoading,
    updateProgress,
    stopLoading
  }
}
