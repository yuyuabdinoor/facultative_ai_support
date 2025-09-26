"use client";
import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";

type ApplicationDetail = {
  id: string;
  status: string;
  created_at: string;
  updated_at: string;
  documents: Document[];
  risk_parameters?: RiskParameters;
  financial_data?: FinancialData;
  risk_analysis?: RiskAnalysis;
  recommendation?: Recommendation;
  analysis_document_data?: AnalysisDocumentData;
};

type Document = {
  id: string;
  filename: string;
  document_type: string;
  processed: boolean;
  upload_timestamp: string;
  metadata?: any;
};

type RiskParameters = {
  asset_value: number;
  coverage_limit: number;
  asset_type: string;
  location: string;
  industry_sector: string;
  construction_type: string;
  occupancy: string;
};

type FinancialData = {
  revenue: number;
  profit_margin: number;
  debt_to_equity: number;
  current_ratio: number;
  financial_strength_score: number;
};

type RiskAnalysis = {
  risk_score: number;
  risk_level: string;
  confidence: number;
  factors: Array<{
    factor: string;
    impact: number;
    description: string;
  }>;
  recommendations: string[];
  analysis_details: {
    loss_history_analysis: any;
    catastrophe_exposure: any;
    financial_strength_assessment: any;
  };
};

type Recommendation = {
  decision: string;
  rationale: string;
  confidence: number;
  conditions?: string[];
  premium_adjustment?: number;
  decision_factors: Array<{
    factor: string;
    weight: number;
    score: number;
  }>;
};

type AnalysisDocumentData = {
  reference_number?: string;
  insured_name?: string;
  cedant_reinsured?: string;
  broker_name?: string;
  perils_covered?: string;
  total_sums_insured?: number;
  currency?: string;
  period_of_insurance?: string;
  pml_percentage?: number;
  share_offered_percentage?: number;
  // ... other 23 fields
};

export default function ApplicationDetailPage() {
  const params = useParams();
  const applicationId = params.id as string;
  const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8005";
  
  const [application, setApplication] = useState<ApplicationDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState("overview");

  const fetchApplication = async () => {
    try {
      const response = await fetch(`${apiBase}/api/v1/applications/${applicationId}`);
      if (response.ok) {
        const data = await response.json();
        setApplication(data);
      }
    } catch (error) {
      console.error('Failed to fetch application:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (applicationId) {
      fetchApplication();
    }
  }, [applicationId]);

  const exportData = async (format: 'json' | 'excel') => {
    try {
      const response = await fetch(`${apiBase}/api/v1/applications/${applicationId}/export?format=${format}`);
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        const filename = format === 'excel' ? `application-${applicationId}.xlsx` : `application-${applicationId}.json`;
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
      }
    } catch (error) {
      console.error('Failed to export data:', error);
    }
  };

  if (loading) {
    return (
      <div className="space-y-4">
        <h1 className="text-2xl font-semibold">Application Details</h1>
        <div className="text-muted-foreground">Loading application...</div>
      </div>
    );
  }

  if (!application) {
    return (
      <div className="space-y-4">
        <h1 className="text-2xl font-semibold">Application Not Found</h1>
        <Link href="/applications" className="text-blue-600 hover:text-blue-800">
          ← Back to Applications
        </Link>
      </div>
    );
  }

  const RiskScoreChart = ({ score, level }: { score: number; level: string }) => {
    const getColor = (level: string) => {
      switch (level) {
        case 'LOW': return 'text-green-600';
        case 'MEDIUM': return 'text-yellow-600';
        case 'HIGH': return 'text-orange-600';
        case 'CRITICAL': return 'text-red-600';
        default: return 'text-gray-600';
      }
    };

    return (
      <div className="flex items-center space-x-4">
        <div className="relative w-24 h-24">
          <svg className="w-24 h-24 transform -rotate-90" viewBox="0 0 100 100">
            <circle
              cx="50"
              cy="50"
              r="40"
              stroke="currentColor"
              strokeWidth="8"
              fill="transparent"
              className="text-gray-200"
            />
            <circle
              cx="50"
              cy="50"
              r="40"
              stroke="currentColor"
              strokeWidth="8"
              fill="transparent"
              strokeDasharray={`${score * 251.2} 251.2`}
              className={getColor(level)}
            />
          </svg>
          <div className="absolute inset-0 flex items-center justify-center">
            <span className={`text-lg font-bold ${getColor(level)}`}>
              {(score * 100).toFixed(0)}%
            </span>
          </div>
        </div>
        <div>
          <div className={`text-lg font-semibold ${getColor(level)}`}>{level}</div>
          <div className="text-sm text-gray-600">Risk Level</div>
        </div>
      </div>
    );
  };

  const FactorChart = ({ factors }: { factors: Array<{factor: string; impact: number; description: string}> }) => (
    <div className="space-y-3">
      {factors.map((factor, index) => (
        <div key={index} className="space-y-1">
          <div className="flex justify-between text-sm">
            <span>{factor.factor}</span>
            <span className="font-medium">{(factor.impact * 100).toFixed(1)}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div 
              className={`h-2 rounded-full ${
                factor.impact > 0.7 ? 'bg-red-500' : 
                factor.impact > 0.4 ? 'bg-yellow-500' : 'bg-green-500'
              }`}
              style={{ width: `${factor.impact * 100}%` }}
            />
          </div>
          <div className="text-xs text-gray-600">{factor.description}</div>
        </div>
      ))}
    </div>
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <Link href="/applications" className="text-blue-600 hover:text-blue-800 text-sm">
            ← Back to Applications
          </Link>
          <h1 className="text-2xl font-semibold mt-2">Application {application.id.slice(0, 8)}</h1>
          <p className="text-gray-600">
            Created: {new Date(application.created_at).toLocaleString()}
          </p>
        </div>
        <div className="flex gap-2">
          <button 
            onClick={() => exportData('json')}
            className="px-3 py-2 border rounded-md hover:bg-gray-50"
          >
            Export JSON
          </button>
          <button 
            onClick={() => exportData('excel')}
            className="px-3 py-2 bg-green-600 text-white rounded-md hover:bg-green-700"
          >
            Export Excel
          </button>
        </div>
      </div>

      {/* Status Banner */}
      <div className={`p-4 rounded-lg ${
        application.status === 'completed' ? 'bg-green-50 border border-green-200' :
        application.status === 'rejected' ? 'bg-red-50 border border-red-200' :
        application.status === 'analyzed' ? 'bg-blue-50 border border-blue-200' :
        'bg-yellow-50 border border-yellow-200'
      }`}>
        <div className="flex items-center justify-between">
          <div>
            <h3 className="font-semibold">Status: {application.status.charAt(0).toUpperCase() + application.status.slice(1)}</h3>
            <p className="text-sm">Last updated: {new Date(application.updated_at).toLocaleString()}</p>
          </div>
          {application.recommendation && (
            <div className={`px-4 py-2 rounded-lg font-semibold ${
              application.recommendation.decision === 'APPROVE' ? 'bg-green-100 text-green-800' :
              application.recommendation.decision === 'REJECT' ? 'bg-red-100 text-red-800' :
              'bg-yellow-100 text-yellow-800'
            }`}>
              {application.recommendation.decision}
            </div>
          )}
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="border-b border-gray-200">
        <nav className="flex space-x-8">
          {[
            { id: 'overview', label: 'Overview' },
            { id: 'documents', label: 'Documents' },
            { id: 'extracted-data', label: 'Extracted Data' },
            { id: 'risk-analysis', label: 'Risk Analysis' },
            { id: 'recommendation', label: 'Recommendation' },
            { id: 'audit-trail', label: 'Audit Trail' }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`py-2 px-1 border-b-2 font-medium text-sm ${
                activeTab === tab.id
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </nav>
      </div>

      {/* Tab Content */}
      <div className="space-y-6">
        {activeTab === 'overview' && (
          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-white rounded-lg border p-6">
              <h3 className="text-lg font-semibold mb-4">Risk Assessment</h3>
              {application.risk_analysis ? (
                <RiskScoreChart 
                  score={application.risk_analysis.risk_score} 
                  level={application.risk_analysis.risk_level} 
                />
              ) : (
                <div className="text-gray-500">No risk analysis available</div>
              )}
            </div>

            <div className="bg-white rounded-lg border p-6">
              <h3 className="text-lg font-semibold mb-4">Key Metrics</h3>
              <div className="space-y-3">
                {application.risk_parameters && (
                  <>
                    <div className="flex justify-between">
                      <span>Asset Value:</span>
                      <span className="font-medium">${application.risk_parameters.asset_value?.toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Coverage Limit:</span>
                      <span className="font-medium">${application.risk_parameters.coverage_limit?.toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Industry:</span>
                      <span className="font-medium">{application.risk_parameters.industry_sector}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Location:</span>
                      <span className="font-medium">{application.risk_parameters.location}</span>
                    </div>
                  </>
                )}
              </div>
            </div>

            <div className="bg-white rounded-lg border p-6">
              <h3 className="text-lg font-semibold mb-4">Documents ({application.documents.length})</h3>
              <div className="space-y-2">
                {application.documents.map(doc => (
                  <div key={doc.id} className="flex items-center justify-between p-2 rounded border">
                    <div className="flex items-center space-x-2">
                      <div className={`w-2 h-2 rounded-full ${doc.processed ? 'bg-green-500' : 'bg-yellow-500'}`} />
                      <span className="text-sm">{doc.filename}</span>
                    </div>
                    <span className="text-xs bg-gray-100 px-2 py-1 rounded">{doc.document_type}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-white rounded-lg border p-6">
              <h3 className="text-lg font-semibold mb-4">Recommendation Summary</h3>
              {application.recommendation ? (
                <div className="space-y-3">
                  <div className={`text-lg font-semibold ${
                    application.recommendation.decision === 'APPROVE' ? 'text-green-600' :
                    application.recommendation.decision === 'REJECT' ? 'text-red-600' :
                    'text-yellow-600'
                  }`}>
                    {application.recommendation.decision}
                  </div>
                  <div className="text-sm">
                    Confidence: {(application.recommendation.confidence * 100).toFixed(1)}%
                  </div>
                  {application.recommendation.premium_adjustment && (
                    <div className="text-sm">
                      Premium Adjustment: {application.recommendation.premium_adjustment > 0 ? '+' : ''}{application.recommendation.premium_adjustment}%
                    </div>
                  )}
                  <div className="text-sm text-gray-600">
                    {application.recommendation.rationale}
                  </div>
                </div>
              ) : (
                <div className="text-gray-500">No recommendation available</div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'documents' && (
          <div className="bg-white rounded-lg border">
            <div className="p-6">
              <h3 className="text-lg font-semibold mb-4">Documents</h3>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-4 py-3 text-left">Filename</th>
                      <th className="px-4 py-3 text-left">Type</th>
                      <th className="px-4 py-3 text-left">Status</th>
                      <th className="px-4 py-3 text-left">Uploaded</th>
                      <th className="px-4 py-3 text-left">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y">
                    {application.documents.map(doc => (
                      <tr key={doc.id}>
                        <td className="px-4 py-3">{doc.filename}</td>
                        <td className="px-4 py-3">
                          <span className="px-2 py-1 bg-gray-100 rounded text-xs">{doc.document_type}</span>
                        </td>
                        <td className="px-4 py-3">
                          <span className={`px-2 py-1 rounded text-xs ${
                            doc.processed ? 'bg-green-100 text-green-800' : 'bg-yellow-100 text-yellow-800'
                          }`}>
                            {doc.processed ? 'Processed' : 'Pending'}
                          </span>
                        </td>
                        <td className="px-4 py-3 text-sm">{new Date(doc.upload_timestamp).toLocaleString()}</td>
                        <td className="px-4 py-3">
                          <a
                            href={`${apiBase}/api/v1/documents/${doc.id}/download`}
                            className="text-blue-600 hover:text-blue-800 text-sm"
                          >
                            Download
                          </a>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'risk-analysis' && application.risk_analysis && (
          <div className="space-y-6">
            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-white rounded-lg border p-6">
                <h3 className="text-lg font-semibold mb-4">Risk Score Breakdown</h3>
                <RiskScoreChart 
                  score={application.risk_analysis.risk_score} 
                  level={application.risk_analysis.risk_level} 
                />
                <div className="mt-4 text-sm text-gray-600">
                  Confidence: {(application.risk_analysis.confidence * 100).toFixed(1)}%
                </div>
              </div>

              <div className="bg-white rounded-lg border p-6">
                <h3 className="text-lg font-semibold mb-4">Risk Factors</h3>
                <FactorChart factors={application.risk_analysis.factors} />
              </div>
            </div>

            <div className="bg-white rounded-lg border p-6">
              <h3 className="text-lg font-semibold mb-4">Analysis Recommendations</h3>
              <ul className="space-y-2">
                {application.risk_analysis.recommendations.map((rec, index) => (
                  <li key={index} className="flex items-start space-x-2">
                    <span className="text-blue-600">•</span>
                    <span className="text-sm">{rec}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        )}

        {activeTab === 'recommendation' && application.recommendation && (
          <div className="space-y-6">
            <div className="bg-white rounded-lg border p-6">
              <h3 className="text-lg font-semibold mb-4">Decision Summary</h3>
              <div className={`text-2xl font-bold mb-2 ${
                application.recommendation.decision === 'APPROVE' ? 'text-green-600' :
                application.recommendation.decision === 'REJECT' ? 'text-red-600' :
                'text-yellow-600'
              }`}>
                {application.recommendation.decision}
              </div>
              <div className="text-sm text-gray-600 mb-4">
                Confidence: {(application.recommendation.confidence * 100).toFixed(1)}%
              </div>
              <p className="text-gray-700">{application.recommendation.rationale}</p>
              
              {application.recommendation.conditions && (
                <div className="mt-4">
                  <h4 className="font-semibold mb-2">Conditions:</h4>
                  <ul className="space-y-1">
                    {application.recommendation.conditions.map((condition, index) => (
                      <li key={index} className="text-sm flex items-start space-x-2">
                        <span className="text-blue-600">•</span>
                        <span>{condition}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>

            {application.recommendation.decision_factors && (
              <div className="bg-white rounded-lg border p-6">
                <h3 className="text-lg font-semibold mb-4">Decision Factors</h3>
                <div className="space-y-3">
                  {application.recommendation.decision_factors.map((factor, index) => (
                    <div key={index} className="space-y-1">
                      <div className="flex justify-between text-sm">
                        <span>{factor.factor}</span>
                        <span className="font-medium">Weight: {(factor.weight * 100).toFixed(1)}% | Score: {factor.score.toFixed(2)}</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div 
                          className="h-2 rounded-full bg-blue-500"
                          style={{ width: `${factor.score * 20}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'extracted-data' && (
          <div className="bg-white rounded-lg border p-6">
            <h3 className="text-lg font-semibold mb-4">Extracted Data</h3>
            {application.analysis_document_data ? (
              <div className="grid md:grid-cols-2 gap-6">
                <div className="space-y-3">
                  <h4 className="font-medium">Basic Information</h4>
                  <div className="space-y-2 text-sm">
                    <div><strong>Reference Number:</strong> {application.analysis_document_data.reference_number || 'N/A'}</div>
                    <div><strong>Insured Name:</strong> {application.analysis_document_data.insured_name || 'N/A'}</div>
                    <div><strong>Cedant/Reinsured:</strong> {application.analysis_document_data.cedant_reinsured || 'N/A'}</div>
                    <div><strong>Broker Name:</strong> {application.analysis_document_data.broker_name || 'N/A'}</div>
                  </div>
                </div>
                <div className="space-y-3">
                  <h4 className="font-medium">Coverage Details</h4>
                  <div className="space-y-2 text-sm">
                    <div><strong>Perils Covered:</strong> {application.analysis_document_data.perils_covered || 'N/A'}</div>
                    <div><strong>Total Sums Insured:</strong> {application.analysis_document_data.total_sums_insured ? `${application.analysis_document_data.currency || ''} ${application.analysis_document_data.total_sums_insured.toLocaleString()}` : 'N/A'}</div>
                    <div><strong>Period of Insurance:</strong> {application.analysis_document_data.period_of_insurance || 'N/A'}</div>
                    <div><strong>PML Percentage:</strong> {application.analysis_document_data.pml_percentage ? `${application.analysis_document_data.pml_percentage}%` : 'N/A'}</div>
                    <div><strong>Share Offered:</strong> {application.analysis_document_data.share_offered_percentage ? `${application.analysis_document_data.share_offered_percentage}%` : 'N/A'}</div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-gray-500">No extracted data available</div>
            )}
          </div>
        )}

        {activeTab === 'audit-trail' && (
          <div className="bg-white rounded-lg border p-6">
            <h3 className="text-lg font-semibold mb-4">Audit Trail</h3>
            <div className="space-y-4">
              <div className="border-l-4 border-blue-500 pl-4">
                <div className="font-medium">Application Created</div>
                <div className="text-sm text-gray-600">{new Date(application.created_at).toLocaleString()}</div>
              </div>
              {application.documents.map(doc => (
                <div key={doc.id} className="border-l-4 border-green-500 pl-4">
                  <div className="font-medium">Document Uploaded: {doc.filename}</div>
                  <div className="text-sm text-gray-600">{new Date(doc.upload_timestamp).toLocaleString()}</div>
                </div>
              ))}
              {application.risk_analysis && (
                <div className="border-l-4 border-yellow-500 pl-4">
                  <div className="font-medium">Risk Analysis Completed</div>
                  <div className="text-sm text-gray-600">Risk Level: {application.risk_analysis.risk_level}</div>
                </div>
              )}
              {application.recommendation && (
                <div className="border-l-4 border-purple-500 pl-4">
                  <div className="font-medium">Recommendation Generated</div>
                  <div className="text-sm text-gray-600">Decision: {application.recommendation.decision}</div>
                </div>
              )}
              <div className="border-l-4 border-gray-500 pl-4">
                <div className="font-medium">Last Updated</div>
                <div className="text-sm text-gray-600">{new Date(application.updated_at).toLocaleString()}</div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
