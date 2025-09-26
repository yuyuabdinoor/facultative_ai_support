"use client";
import { useEffect, useState } from "react";
import Link from "next/link";

type Application = {
  id: string;
  status: string;
  created_at: string;
  updated_at: string;
  documents: Document[];
  risk_analysis?: RiskAnalysis;
  recommendation?: Recommendation;
};

type Document = {
  id: string;
  filename: string;
  document_type: string;
  processed: boolean;
  upload_timestamp: string;
};

type RiskAnalysis = {
  risk_score: number;
  risk_level: string;
  confidence: number;
  factors: string[];
  recommendations: string[];
};

type Recommendation = {
  decision: string;
  rationale: string;
  confidence: number;
  conditions?: string[];
  premium_adjustment?: number;
};

export default function ApplicationsPage() {
  const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8005";
  const [applications, setApplications] = useState<Application[]>([]);
  const [filteredApplications, setFilteredApplications] = useState<Application[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [sortBy, setSortBy] = useState("created_at");
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("desc");

  const fetchApplications = async () => {
    try {
      const response = await fetch(`${apiBase}/api/v1/applications/`);
      if (response.ok) {
        const data = await response.json();
        setApplications(data);
        setFilteredApplications(data);
      }
    } catch (error) {
      console.error('Failed to fetch applications:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchApplications();
  }, []);

  useEffect(() => {
    let filtered = [...applications];

    // Apply search filter
    if (searchTerm) {
      filtered = filtered.filter(app => 
        app.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
        app.documents.some(doc => 
          doc.filename.toLowerCase().includes(searchTerm.toLowerCase())
        )
      );
    }

    // Apply status filter
    if (statusFilter !== "all") {
      filtered = filtered.filter(app => app.status === statusFilter);
    }

    // Apply sorting
    filtered.sort((a, b) => {
      let aValue: any = a[sortBy as keyof Application];
      let bValue: any = b[sortBy as keyof Application];

      if (sortBy === "risk_score") {
        aValue = a.risk_analysis?.risk_score || 0;
        bValue = b.risk_analysis?.risk_score || 0;
      }

      if (typeof aValue === "string") {
        aValue = aValue.toLowerCase();
        bValue = bValue.toLowerCase();
      }

      if (sortOrder === "asc") {
        return aValue > bValue ? 1 : -1;
      } else {
        return aValue < bValue ? 1 : -1;
      }
    });

    setFilteredApplications(filtered);
  }, [applications, searchTerm, statusFilter, sortBy, sortOrder]);

  const getStatusBadge = (status: string) => {
    const colors = {
      pending: "bg-yellow-100 text-yellow-800",
      processing: "bg-blue-100 text-blue-800",
      analyzed: "bg-green-100 text-green-800",
      completed: "bg-purple-100 text-purple-800",
      rejected: "bg-red-100 text-red-800"
    };
    return colors[status as keyof typeof colors] || "bg-gray-100 text-gray-800";
  };

  const getRiskLevelBadge = (level: string) => {
    const colors = {
      LOW: "bg-green-100 text-green-800",
      MEDIUM: "bg-yellow-100 text-yellow-800",
      HIGH: "bg-orange-100 text-orange-800",
      CRITICAL: "bg-red-100 text-red-800"
    };
    return colors[level as keyof typeof colors] || "bg-gray-100 text-gray-800";
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString();
  };

  if (loading) {
    return (
      <div className="space-y-4">
        <h1 className="text-2xl font-semibold">Applications</h1>
        <div className="text-muted-foreground">Loading applications...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold">Applications</h1>
        <button 
          onClick={fetchApplications}
          className="px-3 py-2 border rounded-md hover:bg-gray-50"
        >
          Refresh
        </button>
      </div>

      {/* Filters and Search */}
      <div className="bg-white p-4 rounded-lg border space-y-4">
        <div className="flex flex-wrap gap-4">
          <div className="flex-1 min-w-64">
            <input
              type="text"
              placeholder="Search by ID or document filename..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full px-3 py-2 border rounded-md"
            />
          </div>
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="px-3 py-2 border rounded-md"
          >
            <option value="all">All Status</option>
            <option value="pending">Pending</option>
            <option value="processing">Processing</option>
            <option value="analyzed">Analyzed</option>
            <option value="completed">Completed</option>
            <option value="rejected">Rejected</option>
          </select>
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value)}
            className="px-3 py-2 border rounded-md"
          >
            <option value="created_at">Created Date</option>
            <option value="updated_at">Updated Date</option>
            <option value="status">Status</option>
            <option value="risk_score">Risk Score</option>
          </select>
          <button
            onClick={() => setSortOrder(sortOrder === "asc" ? "desc" : "asc")}
            className="px-3 py-2 border rounded-md hover:bg-gray-50"
          >
            {sortOrder === "asc" ? "↑ Asc" : "↓ Desc"}
          </button>
        </div>
      </div>

      {/* Applications List */}
      {filteredApplications.length === 0 ? (
        <div className="text-center py-8">
          <div className="text-muted-foreground mb-4">No applications found</div>
        </div>
      ) : (
        <div className="grid gap-4">
          {filteredApplications.map((app) => (
            <div key={app.id} className="bg-white rounded-lg border p-6 hover:shadow-md transition-shadow">
              <div className="flex items-start justify-between mb-4">
                <div>
                  <h3 className="text-lg font-semibold">Application {app.id.slice(0, 8)}</h3>
                  <p className="text-sm text-gray-600">
                    Created: {formatDate(app.created_at)} • Updated: {formatDate(app.updated_at)}
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <span className={`px-2 py-1 rounded-full text-xs ${getStatusBadge(app.status)}`}>
                    {app.status.charAt(0).toUpperCase() + app.status.slice(1)}
                  </span>
                  {app.risk_analysis && (
                    <span className={`px-2 py-1 rounded-full text-xs ${getRiskLevelBadge(app.risk_analysis.risk_level)}`}>
                      {app.risk_analysis.risk_level}
                    </span>
                  )}
                </div>
              </div>

              <div className="grid md:grid-cols-3 gap-4 mb-4">
                <div>
                  <h4 className="font-medium text-sm text-gray-700 mb-2">Documents ({app.documents.length})</h4>
                  <div className="space-y-1">
                    {app.documents.slice(0, 3).map((doc) => (
                      <div key={doc.id} className="text-sm">
                        <span className={`inline-block w-2 h-2 rounded-full mr-2 ${doc.processed ? 'bg-green-500' : 'bg-yellow-500'}`} />
                        {doc.filename}
                      </div>
                    ))}
                    {app.documents.length > 3 && (
                      <div className="text-sm text-gray-500">+{app.documents.length - 3} more</div>
                    )}
                  </div>
                </div>

                <div>
                  <h4 className="font-medium text-sm text-gray-700 mb-2">Risk Analysis</h4>
                  {app.risk_analysis ? (
                    <div className="space-y-1">
                      <div className="text-sm">
                        Score: <span className="font-medium">{app.risk_analysis.risk_score.toFixed(2)}</span>
                      </div>
                      <div className="text-sm">
                        Confidence: <span className="font-medium">{(app.risk_analysis.confidence * 100).toFixed(1)}%</span>
                      </div>
                      <div className="text-xs text-gray-600">
                        {app.risk_analysis.factors.length} factors analyzed
                      </div>
                    </div>
                  ) : (
                    <div className="text-sm text-gray-500">Not analyzed</div>
                  )}
                </div>

                <div>
                  <h4 className="font-medium text-sm text-gray-700 mb-2">Recommendation</h4>
                  {app.recommendation ? (
                    <div className="space-y-1">
                      <div className={`text-sm font-medium ${
                        app.recommendation.decision === 'APPROVE' ? 'text-green-600' : 
                        app.recommendation.decision === 'REJECT' ? 'text-red-600' : 'text-yellow-600'
                      }`}>
                        {app.recommendation.decision}
                      </div>
                      <div className="text-sm">
                        Confidence: <span className="font-medium">{(app.recommendation.confidence * 100).toFixed(1)}%</span>
                      </div>
                      {app.recommendation.premium_adjustment && (
                        <div className="text-xs text-gray-600">
                          Premium adj: {app.recommendation.premium_adjustment > 0 ? '+' : ''}{app.recommendation.premium_adjustment}%
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="text-sm text-gray-500">No recommendation</div>
                  )}
                </div>
              </div>

              <div className="flex items-center justify-between pt-4 border-t">
                <div className="text-sm text-gray-500">
                  ID: {app.id}
                </div>
                <Link
                  href={`/applications/${app.id}`}
                  className="px-3 py-1 bg-primary text-primary-foreground rounded text-sm hover:bg-primary/90"
                >
                  View Details
                </Link>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
