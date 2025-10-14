'use client'

import { Fragment } from 'react'
import { Tab } from '@headlessui/react'
import { clsx } from 'clsx'
import { 
  ChartBarIcon, 
  CpuChipIcon, 
  BeakerIcon, 
  DocumentTextIcon, 
  CogIcon, 
  PlayIcon 
} from '@heroicons/react/24/outline'

interface NavigationProps {
  activeTab: string
  onTabChange: (tab: string) => void
}

const tabs = [
  { id: 'data', name: 'Data Ingestion', icon: ChartBarIcon },
  { id: 'optimize', name: 'Portfolio Optimization', icon: CpuChipIcon },
  { id: 'backtest', name: 'Backtesting', icon: BeakerIcon },
  { id: 'report', name: 'Reporting', icon: DocumentTextIcon },
  { id: 'config', name: 'Configuration', icon: CogIcon },
  { id: 'monitor', name: 'Job Monitor', icon: PlayIcon },
]

export function Navigation({ activeTab, onTabChange }: NavigationProps) {
  return (
    <div className="glass-effect border-b border-white/20 shadow-soft">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-20">
          <div className="flex">
            <div className="flex-shrink-0 flex items-center">
              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-xl flex items-center justify-center quantum-glow">
                  <CpuChipIcon className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h1 className="text-2xl font-bold bg-gradient-to-r from-gray-900 to-gray-700 bg-clip-text text-transparent">
                    QEPO
                  </h1>
                  <p className="text-xs text-gray-600 font-medium">Quantum Enhanced Portfolio Optimizer</p>
                </div>
              </div>
            </div>
            <nav className="hidden sm:ml-8 sm:flex sm:space-x-1">
              {tabs.map((tab) => {
                const Icon = tab.icon
                return (
                  <button
                    key={tab.id}
                    onClick={() => onTabChange(tab.id)}
                    className={clsx(
                      'inline-flex items-center px-4 py-2 rounded-lg text-sm font-semibold transition-all duration-200',
                      activeTab === tab.id
                        ? 'bg-gradient-to-r from-cyan-500 to-blue-600 text-white shadow-lg'
                        : 'text-gray-600 hover:text-gray-900 hover:bg-white/50'
                    )}
                  >
                    <Icon className="w-4 h-4 mr-2" />
                    <span className="hidden lg:inline">{tab.name}</span>
                  </button>
                )
              })}
            </nav>
          </div>
        </div>
      </div>
      
      {/* Mobile menu */}
      <div className="sm:hidden border-t border-white/20">
        <div className="pt-2 pb-3 space-y-1 px-4">
          {tabs.map((tab) => {
            const Icon = tab.icon
            return (
              <button
                key={tab.id}
                onClick={() => onTabChange(tab.id)}
                className={clsx(
                  'flex items-center w-full px-3 py-2 rounded-lg text-base font-medium transition-all duration-200',
                  activeTab === tab.id
                    ? 'bg-gradient-to-r from-cyan-500 to-blue-600 text-white'
                    : 'text-gray-600 hover:text-gray-900 hover:bg-white/50'
                )}
              >
                <Icon className="w-5 h-5 mr-3" />
                {tab.name}
              </button>
            )
          })}
        </div>
      </div>
    </div>
  )
}
