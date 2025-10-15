'use client'

import { useState } from 'react'
import { 
  ChartBarIcon, 
  CogIcon, 
  PlayIcon, 
  DocumentTextIcon,
  CpuChipIcon,
  BeakerIcon
} from '@heroicons/react/24/outline'
import { Navigation } from '@/components/Navigation'
import { DataIngestion } from '@/components/DataIngestion'
import { PortfolioOptimization } from '@/components/PortfolioOptimization'
import { Backtesting } from '@/components/Backtesting'
import { Reporting } from '@/components/Reporting'
import { Configuration } from '@/components/Configuration'
import { LoadingScreenDemo } from '@/components/LoadingScreenDemo'
import { JobMonitor } from '@/components/JobMonitor'

type TabType = 'data' | 'optimize' | 'backtest' | 'report' | 'config' | 'monitor' | 'demo'

const tabs = [
  { id: 'data', name: 'Data Ingestion', icon: ChartBarIcon },
  { id: 'optimize', name: 'Portfolio Optimization', icon: CpuChipIcon },
  { id: 'backtest', name: 'Backtesting', icon: BeakerIcon },
  { id: 'report', name: 'Reporting', icon: DocumentTextIcon },
  { id: 'config', name: 'Configuration', icon: CogIcon },
  { id: 'monitor', name: 'Job Monitor', icon: PlayIcon },
  { id: 'demo', name: 'Loading Demo', icon: CpuChipIcon },
]

export default function Home() {
  const [activeTab, setActiveTab] = useState<TabType>('data')

  const renderTabContent = () => {
    switch (activeTab) {
      case 'data':
        return <DataIngestion />
      case 'optimize':
        return <PortfolioOptimization />
      case 'backtest':
        return <Backtesting />
      case 'report':
        return <Reporting />
      case 'config':
        return <Configuration />
      case 'monitor':
        return <JobMonitor />
      case 'demo':
        return <LoadingScreenDemo />
      default:
        return <DataIngestion />
    }
  }

  return (
    <div className="min-h-screen gradient-bg">
      <Navigation activeTab={activeTab} onTabChange={(tab) => setActiveTab(tab as TabType)} />
      
      <main className="max-w-7xl mx-auto py-8 sm:px-6 lg:px-8">
        <div className="px-4 py-6 sm:px-0">
          <div className="glass-effect rounded-2xl p-8 shadow-soft">
            {renderTabContent()}
          </div>
        </div>
      </main>
    </div>
  )
}
