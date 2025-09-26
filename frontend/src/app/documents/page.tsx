"use client";
import { useEffect, useState } from "react";
import Link from "next/link";

type Document = {
  id: string;
  filename: string;
  document_type: string;
  processed: boolean;
  upload_timestamp: string;
  file_path: string;
  metadata?: {
    processing_status?: string;
    processing_task_id?: string;
    ocr_result?: {
      success: boolean;
      confidence: number;
      text?: string;
    };
    email_parsing_result?: {
      subject?: string;
      sender?: string;
      body?: string;
    };
  };
};

type ProcessingStatus = {
  status: string;
  progress?: number;
  message?: string;
  success?: boolean;
  error?: string;
  ocr_result?: {
    success: boolean;
    confidence: number;
    text: string;
  };
};

export default function DocumentsPage() {
  const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8005";
  const [documents, setDocuments] = useState<Document[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [processingStatuses, setProcessingStatuses] = useState<Record<string, ProcessingStatus>>({});

  const fetchDocuments = async () => {
    try {
      const response = await fetch(`${apiBase}/api/v1/documents/`);
      if (!response.ok) {
        throw new Error(`Failed to fetch documents: ${response.status}`);
      }
      const data = await response.json();
      setDocuments(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to fetch documents");
    } finally {
      setLoading(false);
    }
  };

  const fetchProcessingStatus = async (documentId: string) => {
    try {
      const response = await fetch(`${apiBase}/api/v1/documents/${documentId}/ocr/status`);
      if (response.ok) {
        const status = await response.json();
        setProcessingStatuses(prev => ({
          ...prev,
          [documentId]: status
        }));
      }
    } catch (err) {
      console.error(`Failed to fetch processing status for ${documentId}:`, err);
    }
  };

  const viewExtractedText = async (documentId: string) => {
    try {
      const res = await fetch(`${apiBase}/api/v1/documents/${documentId}/ocr/text`);
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err?.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      alert(`Extracted text (${data.confidence}% confidence):\n\n${(data.text || '').slice(0, 500)}${(data.text || '').length > 500 ? '...' : ''}`);
    } catch (e: any) {
      alert(`Failed to fetch extracted text: ${e.message || e}`);
    }
  };

  const triggerOCR = async (documentId: string) => {
    try {
      const response = await fetch(`${apiBase}/api/v1/documents/${documentId}/ocr`, {
        method: 'POST'
      });
      if (response.ok) {
        const result = await response.json();
        // Start polling for this document
        fetchProcessingStatus(documentId);
        // Show success message
        alert('OCR processing triggered successfully');
      } else {
        const error = await response.json();
        alert(`Failed to trigger OCR: ${error.detail || 'Unknown error'}`);
      }
    } catch (err) {
      alert(`Error triggering OCR: ${err instanceof Error ? err.message : 'Unknown error'}`);
    }
  };

  // Poll processing status for documents that are being processed
  useEffect(() => {
    const interval = setInterval(() => {
      documents.forEach(doc => {
        const status = processingStatuses[doc.id];
        if (status && (status.status === 'processing' || status.status === 'queued')) {
          fetchProcessingStatus(doc.id);
        }
      });
    }, 3000); // Poll every 3 seconds

    return () => clearInterval(interval);
  }, [documents, processingStatuses]);

  useEffect(() => {
    fetchDocuments();
  }, []);

  const getStatusBadge = (doc: Document) => {
    const status = processingStatuses[doc.id];
    
    if (status) {
      switch (status.status) {
        case 'completed':
          return <span className="px-2 py-1 bg-green-100 text-green-800 rounded-full text-xs">Processed</span>;
        case 'processing':
          return <span className="px-2 py-1 bg-blue-100 text-blue-800 rounded-full text-xs">Processing...</span>;
        case 'queued':
          return <span className="px-2 py-1 bg-yellow-100 text-yellow-800 rounded-full text-xs">Queued</span>;
        case 'failed':
          return <span className="px-2 py-1 bg-red-100 text-red-800 rounded-full text-xs">Failed</span>;
        default:
          return <span className="px-2 py-1 bg-gray-100 text-gray-800 rounded-full text-xs">Unknown</span>;
      }
    }
    
    if (doc.processed) {
      return <span className="px-2 py-1 bg-green-100 text-green-800 rounded-full text-xs">Processed</span>;
    }
    
    return <span className="px-2 py-1 bg-gray-100 text-gray-800 rounded-full text-xs">Not Processed</span>;
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  const formatFileSize = (path: string) => {
    // This is a placeholder - in a real app you'd want to get actual file size
    return "N/A KB";
  };

  if (loading) {
    return (
      <div className="space-y-4">
        <h1 className="text-2xl font-semibold">Documents</h1>
        <div className="flex items-center justify-center py-8">
          <div className="text-muted-foreground">Loading documents...</div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="space-y-4">
        <h1 className="text-2xl font-semibold">Documents</h1>
        <div className="bg-red-50 border border-red-200 rounded-md p-4">
          <div className="text-red-800">Error: {error}</div>
          <button 
            onClick={fetchDocuments}
            className="mt-2 px-3 py-1 bg-red-100 text-red-800 rounded text-sm hover:bg-red-200"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold">Documents</h1>
        <div className="flex gap-2">
          <button 
            onClick={fetchDocuments}
            className="px-3 py-2 border rounded-md hover:bg-gray-50"
          >
            Refresh
          </button>
          <Link 
            href="/upload"
            className="px-3 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90"
          >
            Upload Documents
          </Link>
        </div>
      </div>

      {documents.length === 0 ? (
        <div className="text-center py-8">
          <div className="text-muted-foreground mb-4">No documents found</div>
          <Link 
            href="/upload"
            className="inline-flex px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90"
          >
            Upload Your First Document
          </Link>
        </div>
      ) : (
        <div className="bg-white rounded-lg border overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-50">
                <tr className="text-left">
                  <th className="px-4 py-3 font-medium text-gray-900">Filename</th>
                  <th className="px-4 py-3 font-medium text-gray-900">Type</th>
                  <th className="px-4 py-3 font-medium text-gray-900">Status</th>
                  <th className="px-4 py-3 font-medium text-gray-900">Uploaded</th>
                  <th className="px-4 py-3 font-medium text-gray-900">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {documents.map((doc) => {
                  const status = processingStatuses[doc.id];
                  
                  return (
                    <tr key={doc.id} className="hover:bg-gray-50">
                      <td className="px-4 py-3">
                        <div className="font-medium text-gray-900">{doc.filename}</div>
                        <div className="text-sm text-gray-500 flex items-center gap-2">
                          <span>ID: {doc.id.slice(0, 8)}...</span>
                          {/* Inline OCR status marker */}
                          {(() => {
                            const s = processingStatuses[doc.id];
                            if (!s?.status) return null;
                            const color = s.status === 'completed' ? 'bg-green-500' : s.status === 'processing' ? 'bg-blue-500' : s.status === 'queued' ? 'bg-yellow-500' : s.status === 'failed' ? 'bg-red-500' : 'bg-gray-400';
                            return (
                              <span className="inline-flex items-center gap-1 text-xs" title={`OCR ${s.status}`}>
                                <span className={`inline-block h-2 w-2 rounded-full ${color}`} aria-hidden />
                                <span className="text-gray-600 capitalize">{s.status}</span>
                              </span>
                            );
                          })()}
                          {/* Quick link to text when available */}
                          {(doc.processed || processingStatuses[doc.id]?.ocr_result) && (
                            <button
                              onClick={() => viewExtractedText(doc.id)}
                              className="text-xs text-blue-700 hover:underline"
                              title="View extracted text"
                            >
                              Text
                            </button>
                          )}
                        </div>
                      </td>
                      <td className="px-4 py-3">
                        <span className="px-2 py-1 bg-gray-100 text-gray-800 rounded text-xs uppercase">
                          {doc.document_type}
                        </span>
                      </td>
                      <td className="px-4 py-3">
                        <div className="space-y-1">
                          {getStatusBadge(doc)}
                          {status?.progress && (
                            <div className="w-full bg-gray-200 rounded-full h-1.5">
                              <div 
                                className="bg-blue-600 h-1.5 rounded-full transition-all duration-300"
                                style={{ width: `${status.progress}%` }}
                              />
                            </div>
                          )}
                          {status?.message && (
                            <div className="text-xs text-gray-600">{status.message}</div>
                          )}
                        </div>
                      </td>
                      <td className="px-4 py-3 text-sm text-gray-600">
                        {formatDate(doc.upload_timestamp)}
                      </td>
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-2">
                          {!doc.processed && (!status || status.status === 'failed') && (
                            <button
                              onClick={() => triggerOCR(doc.id)}
                              className="px-2 py-1 bg-blue-100 text-blue-800 rounded text-xs hover:bg-blue-200"
                            >
                              Process
                            </button>
                          )}
                          <a
                            href={`${apiBase}/api/v1/documents/${doc.id}/download`}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="px-2 py-1 bg-gray-100 text-gray-800 rounded text-xs hover:bg-gray-200"
                          >
                            Download
                          </a>
                          {(doc.processed || status?.ocr_result) && (
                            <button
                              onClick={() => {
                                // Show extracted text in a modal or navigate to a detail page
                                fetch(`${apiBase}/api/v1/documents/${doc.id}/ocr/text`)
                                  .then(res => res.json())
                                  .then(data => {
                                    alert(`Extracted text (${data.confidence}% confidence):\n\n${data.text?.slice(0, 500)}${data.text?.length > 500 ? '...' : ''}`);
                                  })
                                  .catch(err => alert('Failed to fetch extracted text'));
                              }}
                              className="px-2 py-1 bg-green-100 text-green-800 rounded text-xs hover:bg-green-200"
                            >
                              View Text
                            </button>
                          )}
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
