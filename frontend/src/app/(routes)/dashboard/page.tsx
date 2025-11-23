"use client";

import React, { useState } from 'react';
import { Shield, LayoutDashboard, Activity, BarChart2, Target } from 'lucide-react';
import Overview from '@/components/dashboard/Overview';
import LiveMonitor from '@/components/dashboard/LiveMonitor';
import Analytics from '@/components/dashboard/Analytics';
import ManualPrediction from '@/components/dashboard/ManualPrediction';
import { cn } from '@/lib/utils';
import { DashboardProvider } from '@/contexts/DashboardContext';
import { AnalyticsProvider } from '@/contexts/AnalyticsContext';
import { MonitorProvider } from '@/contexts/MonitorContext';

// Navigation Items
const NAV_ITEMS = [
  { id: 'overview', label: 'Dashboard', icon: LayoutDashboard },
  { id: 'monitor', label: 'Live Monitor', icon: Activity },
  { id: 'analytics', label: 'Analytics', icon: BarChart2 },
];

export default function DashboardPage() {
  const [activeTab, setActiveTab] = useState('overview');

  return (
    
          <div className="min-h-screen bg-[#0a0e27] text-white overflow-hidden font-sans mt-[60px]">
        {/* Background Effects (copied from landing) */}
        <div className="fixed inset-0 z-0 pointer-events-none">
          <div className="absolute inset-0 bg-gradient-to-br from-[#0a0e27] via-[#1a1534] to-[#2d1b3d]"></div>
          <div className="absolute top-0 left-1/4 w-96 h-96 bg-blue-600/20 rounded-full blur-3xl animate-pulse"></div>
          <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-purple-600/20 rounded-full blur-3xl animate-pulse" style={{animationDelay: '1s'}}></div>
        </div>

        {/* Main Content */}
        <div className="relative z-10 flex flex-col h-screen">
          {/* Header */}
          {/* <header className="border-b border-white/10 bg-black/20 backdrop-blur-md">
            <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <div className="bg-gradient-to-br from-blue-600 to-purple-600 p-2 rounded-lg">
                  <Shield className="h-6 w-6 text-white" />
                </div>
                <span className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-400">
                  SkyFort IDS
                </span>
              </div>
              <div className="flex items-center space-x-4">
                <div className="flex items-center space-x-2 px-3 py-1 rounded-full bg-green-500/10 border border-green-500/20">
                  <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
                  <span className="text-xs font-medium text-green-400">System Online</span>
                </div>
                <div className="h-8 w-8 rounded-full bg-gradient-to-br from-gray-700 to-gray-600 border border-white/10"></div>
              </div>
            </div>
          </header> */}

          {/* Dashboard Content */}
          <main className="flex-1 overflow-hidden flex flex-col max-w-7xl mx-auto w-full px-6 py-8">
            {/* Tabs */}
            <div className="flex space-x-1 bg-white/5 p-1 rounded-xl mb-8 w-fit backdrop-blur-sm border border-white/10">
              {NAV_ITEMS.map((item) => (
                <button
                  key={item.id}
                  onClick={() => setActiveTab(item.id)}
                  className={cn(
                    "flex items-center px-6 py-2.5 rounded-lg text-sm font-medium transition-all duration-200",
                    activeTab === item.id
                      ? "bg-gradient-to-r from-blue-600 to-purple-600 text-white shadow-lg"
                      : "text-gray-400 hover:text-white hover:bg-white/5"
                  )}
                >
                  <item.icon className="mr-2 h-4 w-4" />
                  {item.label}
                </button>
              ))}
            </div>

            {/* Tab Content */}
            <div className="flex-1 overflow-y-auto custom-scrollbar pb-10">
              {activeTab === 'overview' && <Overview />}
              {activeTab === 'monitor' && <LiveMonitor />}
              {activeTab === 'analytics' && <Analytics />}
              {activeTab === 'manual' && <ManualPrediction />}
            </div>
          </main>
        </div>
      </div>
        
  );
}
