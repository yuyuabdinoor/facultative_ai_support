"use client";
import { useEffect, useState } from "react";
import Link from "next/link";

type Document = {
  id: string;
  filename: string;
  document_type: string;
  processed: boolean;
  upload_timestamp: string;
};

type DashboardStats = {
  total_documents: number;
  processed_documents: number;
  pending_documents: number;
  recent_uploads: Document[];
};

export default function DashboardPage() {
  const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8005";
  const [stats, setStats] = useState<DashboardStats>({
    total_documents: 0,
    processed_documents: 0,
    pending_documents: 0,
    recent_uploads: []
  });
  const [loading, setLoading] = useState(true);

  const fetchDashboardData = async () => {
    try {
      // Fetch recent documents to calculate stats
      const response = await fetch(`${apiBase}/api/v1/documents/?limit=10`);
      if (response.ok) {
        const documents: Document[] = await response.json();
        
        const total = documents.length;
        const processed = documents.filter(d => d.processed).length;
        const pending = total - processed;
        
        setStats({
          total_documents: total,
          processed_documents: processed,
          pending_documents: pending,
          recent_uploads: documents.slice(0, 5) // Show 5 most recent
        });
      }
    } catch (error) {
      console.error('Failed to fetch dashboard data:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDashboardData();
    
    // Refresh every 30 seconds
    const interval = setInterval(fetchDashboardData, 30000);
    return () => clearInterval(interval);
  }, []);

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString();
  };

  if (loading) {
    return (
      <div className="space-y-4">
        <h1 className="text-2xl font-semibold">Dashboard</h1>
        <div className="text-muted-foreground">Loading dashboard data...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold">Dashboard</h1>
      
      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-white p-6 rounded-lg border">
          <div className="text-2xl font-bold text-blue-600">{stats.total_documents}</div>
          <div className="text-sm text-gray-600">Total Documents</div>
        </div>
        <div className="bg-white p-6 rounded-lg border">
          <div className="text-2xl font-bold text-green-600">{stats.processed_documents}</div>
          <div className="text-sm text-gray-600">Processed</div>
        </div>
        <div className="bg-white p-6 rounded-lg border">
          <div className="text-2xl font-bold text-orange-600">{stats.pending_documents}</div>
          <div className="text-sm text-gray-600">Pending Processing</div>
        </div>
      </div>

      {/* Recent Activity */}
      <div className="bg-white rounded-lg border">
        <div className="p-6 border-b">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold">Recent Uploads</h2>
            <Link 
              href="/documents" 
              className="text-sm text-blue-600 hover:text-blue-800"
            >
              View All
            </Link>
          </div>
        </div>
        <div className="p-6">
          {stats.recent_uploads.length === 0 ? (
            <div className="text-center py-8">
              <div className="text-muted-foreground mb-4">No documents uploaded yet</div>
              <Link 
                href="/upload"
                className="inline-flex px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90"
              >
                Upload Your First Document
              </Link>
            </div>
          ) : (
            <div className="space-y-3">
              {stats.recent_uploads.map((doc) => (
                <div key={doc.id} className="flex items-center justify-between p-3 rounded border">
                  <div className="flex items-center space-x-3">
                    <div className={`w-2 h-2 rounded-full ${doc.processed ? 'bg-green-500' : 'bg-orange-500'}`} />
                    <div>
                      <div className="font-medium">{doc.filename}</div>
                      <div className="text-sm text-gray-500">
                        {doc.document_type.toUpperCase()} • {formatDate(doc.upload_timestamp)}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className={`px-2 py-1 rounded-full text-xs ${
                      doc.processed 
                        ? 'bg-green-100 text-green-800' 
                        : 'bg-orange-100 text-orange-800'
                    }`}>
                      {doc.processed ? 'Processed' : 'Pending'}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Quick Actions */}
      <div className="bg-white rounded-lg border p-6">
        <h2 className="text-lg font-semibold mb-4">Quick Actions</h2>
        <div className="flex flex-wrap gap-3">
          <Link 
            href="/upload"
            className="px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90"
          >
            Upload Documents
          </Link>
          <Link 
            href="/documents"
            className="px-4 py-2 border rounded-md hover:bg-gray-50"
          >
            View All Documents
          </Link>
          <button 
            onClick={fetchDashboardData}
            className="px-4 py-2 border rounded-md hover:bg-gray-50"
          >
            Refresh Data
          </button>
        </div>
      </div>
    </div>
  );
}
