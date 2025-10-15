'use client'

import { useState } from 'react'

export function LinkedInLogo() {
  const [isHovered, setIsHovered] = useState(false)

  return (
    <a
      href="https://www.linkedin.com/in/udhaya-kumar-a-exe/"
      target="_blank"
      rel="noopener noreferrer"
      className="fixed bottom-6 right-6 z-50 group"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      title="Built by Udhaya Kumar A"
    >
      <div className={`
        w-12 h-12 rounded-full bg-gradient-to-br from-blue-600 to-blue-700 
        flex items-center justify-center shadow-lg transition-all duration-300
        hover:shadow-xl hover:scale-110 hover:from-blue-500 hover:to-blue-600
        ${isHovered ? 'shadow-blue-500/50' : 'shadow-gray-900/20'}
      `}>
        <svg
          className="w-6 h-6 text-white transition-transform duration-300 group-hover:scale-110"
          fill="currentColor"
          viewBox="0 0 24 24"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
        </svg>
      </div>
      
      {/* Tooltip */}
      <div className={`
        absolute bottom-full right-0 mb-2 px-3 py-1 bg-gray-900 text-white text-xs rounded-lg
        opacity-0 group-hover:opacity-100 transition-opacity duration-300 whitespace-nowrap
        pointer-events-none
      `}>
        Built by Udhaya Kumar A
        <div className="absolute top-full right-4 w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-gray-900"></div>
      </div>
    </a>
  )
}

