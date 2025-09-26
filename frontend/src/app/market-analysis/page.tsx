"use client";
import { useEffect, useState } from "react";
import { ChevronDownIcon, ChevronUpIcon } from "@heroicons/react/24/outline";

type MarketData = {
  market_groups: MarketGroup[];
  trend_analysis: TrendAnalysis[];
  statistics: MarketStatistics;
};

type MarketGroup = {
  id: string;
  name: string;
  region: string;
  industry_sector: string;
  risk_level: string;
  total_applications: number;
  approval_rate: number;
  average_premium: number;
  market_share: number;
  growth_rate: number;
  applications: Application[];
};

type Application = {
  id: string;
  insured_name: string;
  asset_value: number;
  risk_score: number;
  status: string;
  created_at: string;
  recommendation?: {
    decision: string;
    premium_adjustment: number;
  };
};

type TrendAnalysis = {
  period: string;
  total_applications: number;
  approval_rate: number;
  average_risk_score: number;
  premium_volume: number;
  market_concentration: number;
};

type MarketStatistics = {
  total_markets: number;
  total_applications: number;
  overall_approval_rate: number;
  total_premium_volume: number;
  risk_distribution: {
    low: number;
    medium: number;
    high: number;
    critical: number;
  };
  top_performing_markets: string[];
  emerging_markets: string[];
};

export default function MarketAnalysisPage() {
  const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8005";
  const [marketData, setMarketData] = useState<MarketData | null>(null);
  const [loading, setLoading] = useState(true);
  const [filters, setFilters] = useState({
    region: '',
    industry: '',
    risk_level: '',
    date_range: '6m'
  });
  const [expandedGroup, setExpandedGroup] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'grid' | 'table'>('grid');

  const fetchMarketData = async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams();
      if (filters.region) params.append('region', filters.region);
      if (filters.industry) params.append('industry', filters.industry);
      if (filters.risk_level) params.append('risk_level', filters.risk_level);
      params.append('date_range', filters.date_range);

      const response = await fetch(`${apiBase}/api/v1/market-analysis?${params}`);
      if (response.ok) {
        const data = await response.json();
        setMarketData(data);
      }
    } catch (error) {
      console.error('Failed to fetch market data:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMarketData();
  }, [filters]);

  const exportMarketReport = async (format: 'excel' | 'pdf') => {
    try {
      const params = new URLSearchParams();
      Object.entries(filters).forEach(([key, value]) => {
        if (value) params.append(key, value);
      });
      params.append('format', format);

      const response = await fetch(`${apiBase}/api/v1/market-analysis/export?${params}`);
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        const filename = `market-analysis-${new Date().toISOString().split('T')[0]}.${format}`;
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
      }
    } catch (error) {
      console.error('Failed to export market report:', error);
    }
  };

  if (loading) {
    return (
      <div className="space-y-4">
        <h1 className="text-2xl font-semibold">Market Analysis</h1>
        <div className="text-muted-foreground">Loading market data...</div>
      </div>
    );
  }

  if (!marketData) {
    return (
      <div className="space-y-4">
        <h1 className="text-2xl font-semibold">Market Analysis</h1>
        <div className="text-muted-foreground">Failed to load market data</div>
      </div>
    );
  }

  const TrendChart = ({ trends }: { trends: TrendAnalysis[] }) => {
    const maxVolume = Math.max(...trends.map(t => t.premium_volume));
    
    return (
      <div className="space-y-4">
        <div className="flex justify-between items-center">
          <h4 className="font-medium">Premium Volume Trend</h4>
          <div className="text-sm text-gray-600">Last {trends.length} periods</div>
        </div>
        <div className="relative h-48">
          <div className="absolute inset-0 flex items-end justify-between space-x-1">
            {trends.map((trend, index) => (
              <div key={index} className="flex-1 flex flex-col items-center">
                <div
                  className="w-full bg-blue-500 rounded-t"
                  style={{ height: `${(trend.premium_volume / maxVolume) * 100}%` }}
                  title={`${trend.period}: $${trend.premium_volume.toLocaleString()}`}
                />
                <div className="text-xs mt-1 text-center">{trend.period}</div>
              </div>
            ))}
          </div>
        </div>
        <div className="grid grid-cols-3 gap-4 text-sm">
          <div className="text-center">
            <div className="font-medium">{trends[trends.length - 1]?.total_applications || 0}</div>
            <div className="text-gray-600">Applications</div>
          </div>
          <div className="text-center">
            <div className="font-medium">{((trends[trends.length - 1]?.approval_rate || 0) * 100).toFixed(1)}%</div>
            <div className="text-gray-600">Approval Rate</div>
          </div>
          <div className="text-center">
            <div className="font-medium">${(trends[trends.length - 1]?.premium_volume || 0).toLocaleString()}</div>
            <div className="text-gray-600">Premium Volume</div>
          </div>
        </div>
      </div>
    );
  };

  const RiskDistributionChart = ({ distribution }: { distribution: any }) => {
    const total = Object.values(distribution).reduce((sum: number, value: any) => sum + value, 0);
    
    return (
      <div className="space-y-4">
        <h4 className="font-medium">Risk Distribution</h4>
        <div className="space-y-2">
          {Object.entries(distribution).map(([level, count]: [string, any]) => {
            const percentage = total > 0 ? (count / total) * 100 : 0;
            const color = {
              low: 'bg-green-500',
              medium: 'bg-yellow-500',
              high: 'bg-orange-500',
              critical: 'bg-red-500'
            }[level] || 'bg-gray-500';
            
            return (
              <div key={level} className="space-y-1">
                <div className="flex justify-between text-sm">
                  <span className="capitalize">{level}</span>
                  <span>{count} ({percentage.toFixed(1)}%)</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div className={`h-2 rounded-full ${color}`} style={{ width: `${percentage}%` }} />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Market Analysis & Reporting</h1>
          <p className="text-gray-600">Comprehensive market insights and trend analysis</p>
        </div>
        <div className="flex gap-2">
          <button 
            onClick={() => exportMarketReport('excel')}
            className="px-3 py-2 border rounded-md hover:bg-gray-50"
          >
            Export Excel
          </button>
          <button 
            onClick={() => exportMarketReport('pdf')}
            className="px-3 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
          >
            Export PDF
          </button>
        </div>
      </div>

      {/* Filters */}
      <div className="bg-white rounded-lg border p-4">
        <div className="grid md:grid-cols-4 gap-4">
          <div>
            <label className="block text-sm font-medium mb-1">Region</label>
            <select 
              value={filters.region}
              onChange={(e) => setFilters({...filters, region: e.target.value})}
              className="w-full p-2 border rounded-md"
            >
              <option value="">All Regions</option>
              <option value="North America">North America</option>
              <option value="Europe">Europe</option>
              <option value="Asia Pacific">Asia Pacific</option>
              <option value="Latin America">Latin America</option>
              <option value="Middle East & Africa">Middle East & Africa</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">Industry</label>
            <select 
              value={filters.industry}
              onChange={(e) => setFilters({...filters, industry: e.target.value})}
              className="w-full p-2 border rounded-md"
            >
              <option value="">All Industries</option>
              <option value="Manufacturing">Manufacturing</option>
              <option value="Energy">Energy</option>
              <option value="Technology">Technology</option>
              <option value="Healthcare">Healthcare</option>
              <option value="Financial Services">Financial Services</option>
              <option value="Real Estate">Real Estate</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">Risk Level</label>
            <select 
              value={filters.risk_level}
              onChange={(e) => setFilters({...filters, risk_level: e.target.value})}
              className="w-full p-2 border rounded-md"
            >
              <option value="">All Risk Levels</option>
              <option value="LOW">Low</option>
              <option value="MEDIUM">Medium</option>
              <option value="HIGH">High</option>
              <option value="CRITICAL">Critical</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">Time Period</label>
            <select 
              value={filters.date_range}
              onChange={(e) => setFilters({...filters, date_range: e.target.value})}
              className="w-full p-2 border rounded-md"
            >
              <option value="1m">Last Month</option>
              <option value="3m">Last 3 Months</option>
              <option value="6m">Last 6 Months</option>
              <option value="1y">Last Year</option>
              <option value="all">All Time</option>
            </select>
          </div>
        </div>
      </div>

      {/* Summary Statistics */}
      <div className="grid md:grid-cols-4 gap-4">
        <div className="bg-white rounded-lg border p-6">
          <div className="text-2xl font-bold text-blue-600">{marketData.statistics.total_markets}</div>
          <div className="text-sm text-gray-600">Active Markets</div>
        </div>
        <div className="bg-white rounded-lg border p-6">
          <div className="text-2xl font-bold text-green-600">{marketData.statistics.total_applications}</div>
          <div className="text-sm text-gray-600">Total Applications</div>
        </div>
        <div className="bg-white rounded-lg border p-6">
          <div className="text-2xl font-bold text-purple-600">
            {(marketData.statistics.overall_approval_rate * 100).toFixed(1)}%
          </div>
          <div className="text-sm text-gray-600">Overall Approval Rate</div>
        </div>
        <div className="bg-white rounded-lg border p-6">
          <div className="text-2xl font-bold text-orange-600">
            ${marketData.statistics.total_premium_volume.toLocaleString()}
          </div>
          <div className="text-sm text-gray-600">Premium Volume</div>
        </div>
      </div>

      {/* Charts */}
      <div className="grid md:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg border p-6">
          <TrendChart trends={marketData.trend_analysis} />
        </div>
        <div className="bg-white rounded-lg border p-6">
          <RiskDistributionChart distribution={marketData.statistics.risk_distribution} />
        </div>
      </div>

      {/* Market Performance Insights */}
      <div className="grid md:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg border p-6">
          <h3 className="text-lg font-semibold mb-4">Top Performing Markets</h3>
          <div className="space-y-2">
            {marketData.statistics.top_performing_markets.map((market, index) => (
              <div key={index} className="flex items-center justify-between p-2 rounded bg-green-50">
                <span className="font-medium">#{index + 1}</span>
                <span>{market}</span>
                <span className="text-green-600 text-sm">High Performance</span>
              </div>
            ))}
          </div>
        </div>
        <div className="bg-white rounded-lg border p-6">
          <h3 className="text-lg font-semibold mb-4">Emerging Markets</h3>
          <div className="space-y-2">
            {marketData.statistics.emerging_markets.map((market, index) => (
              <div key={index} className="flex items-center justify-between p-2 rounded bg-blue-50">
                <span className="font-medium">#{index + 1}</span>
                <span>{market}</span>
                <span className="text-blue-600 text-sm">Growing</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* View Toggle */}
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold">Market Groups ({marketData.market_groups.length})</h3>
        <div className="flex rounded-lg border">
          <button
            onClick={() => setViewMode('grid')}
            className={`px-3 py-2 text-sm ${viewMode === 'grid' ? 'bg-blue-600 text-white' : 'hover:bg-gray-50'}`}
          >
            Grid View
          </button>
          <button
            onClick={() => setViewMode('table')}
            className={`px-3 py-2 text-sm ${viewMode === 'table' ? 'bg-blue-600 text-white' : 'hover:bg-gray-50'}`}
          >
            Table View
          </button>
        </div>
      </div>

      {/* Market Groups */}
      {viewMode === 'grid' ? (
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
          {marketData.market_groups.map((group) => (
            <div key={group.id} className="bg-white rounded-lg border hover:shadow-md transition-shadow">
              <div className="p-4">
                <div className="flex items-start justify-between mb-3">
                  <div>
                    <h4 className="font-semibold">{group.name}</h4>
                    <p className="text-sm text-gray-600">{group.region}</p>
                  </div>
                  <span className={`px-2 py-1 rounded text-xs ${
                    group.risk_level === 'LOW' ? 'bg-green-100 text-green-800' :
                    group.risk_level === 'MEDIUM' ? 'bg-yellow-100 text-yellow-800' :
                    group.risk_level === 'HIGH' ? 'bg-orange-100 text-orange-800' :
                    'bg-red-100 text-red-800'
                  }`}>
                    {group.risk_level}
                  </span>
                </div>
                
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span>Applications:</span>
                    <span className="font-medium">{group.total_applications}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Approval Rate:</span>
                    <span className="font-medium">{(group.approval_rate * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Avg Premium:</span>
                    <span className="font-medium">${group.average_premium.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Market Share:</span>
                    <span className="font-medium">{(group.market_share * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Growth Rate:</span>
                    <span className={`font-medium ${group.growth_rate >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                      {group.growth_rate >= 0 ? '+' : ''}{(group.growth_rate * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>

                <button
                  onClick={() => setExpandedGroup(expandedGroup === group.id ? null : group.id)}
                  className="mt-3 w-full px-3 py-2 text-sm border rounded hover:bg-gray-50 flex items-center justify-center space-x-1"
                >
                  <span>View Applications</span>
                  {expandedGroup === group.id ? 
                    <ChevronUpIcon className="w-4 h-4" /> : 
                    <ChevronDownIcon className="w-4 h-4" />
                  }
                </button>

                {expandedGroup === group.id && (
                  <div className="mt-3 space-y-2 max-h-48 overflow-y-auto">
                    {group.applications.map((app) => (
                      <div key={app.id} className="p-2 border rounded text-xs">
                        <div className="flex justify-between items-start">
                          <div>
                            <div className="font-medium">{app.insured_name}</div>
                            <div className="text-gray-600">${app.asset_value.toLocaleString()}</div>
                          </div>
                          <div className="text-right">
                            <div className={`px-1 py-0.5 rounded text-xs ${
                              app.status === 'approved' ? 'bg-green-100 text-green-800' :
                              app.status === 'rejected' ? 'bg-red-100 text-red-800' :
                              'bg-yellow-100 text-yellow-800'
                            }`}>
                              {app.status}
                            </div>
                            <div className="text-gray-600 mt-1">Risk: {(app.risk_score * 10).toFixed(1)}/10</div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="bg-white rounded-lg border overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-3 text-left">Market Group</th>
                  <th className="px-4 py-3 text-left">Region</th>
                  <th className="px-4 py-3 text-left">Industry</th>
                  <th className="px-4 py-3 text-left">Risk Level</th>
                  <th className="px-4 py-3 text-left">Applications</th>
                  <th className="px-4 py-3 text-left">Approval Rate</th>
                  <th className="px-4 py-3 text-left">Avg Premium</th>
                  <th className="px-4 py-3 text-left">Market Share</th>
                  <th className="px-4 py-3 text-left">Growth Rate</th>
                </tr>
              </thead>
              <tbody className="divide-y">
                {marketData.market_groups.map((group) => (
                  <tr key={group.id} className="hover:bg-gray-50">
                    <td className="px-4 py-3 font-medium">{group.name}</td>
                    <td className="px-4 py-3">{group.region}</td>
                    <td className="px-4 py-3">{group.industry_sector}</td>
                    <td className="px-4 py-3">
                      <span className={`px-2 py-1 rounded text-xs ${
                        group.risk_level === 'LOW' ? 'bg-green-100 text-green-800' :
                        group.risk_level === 'MEDIUM' ? 'bg-yellow-100 text-yellow-800' :
                        group.risk_level === 'HIGH' ? 'bg-orange-100 text-orange-800' :
                        'bg-red-100 text-red-800'
                      }`}>
                        {group.risk_level}
                      </span>
                    </td>
                    <td className="px-4 py-3">{group.total_applications}</td>
                    <td className="px-4 py-3">{(group.approval_rate * 100).toFixed(1)}%</td>
                    <td className="px-4 py-3">${group.average_premium.toLocaleString()}</td>
                    <td className="px-4 py-3">{(group.market_share * 100).toFixed(1)}%</td>
                    <td className="px-4 py-3">
                      <span className={group.growth_rate >= 0 ? 'text-green-600' : 'text-red-600'}>
                        {group.growth_rate >= 0 ? '+' : ''}{(group.growth_rate * 100).toFixed(1)}%
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
