"use client";
import { useState } from "react";

export default function DemoOrchestrator() {
  const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8005";

  const [applicationId, setApplicationId] = useState<string>("");
  const [docIds, setDocIds] = useState<string[]>([]);
  const [log, setLog] = useState<string>("");
  const [uploading, setUploading] = useState(false);

  const appendLog = (msg: string) => setLog((l) => `${new Date().toLocaleTimeString()} - ${msg}\n` + l);

  const createApplication = async () => {
    try {
      const res = await fetch(`${apiBase}/api/v1/applications/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.detail || res.statusText);
      setApplicationId(data.id);
      appendLog(`Created application ${data.id}`);
    } catch (e: any) {
      appendLog(`Create application failed: ${e.message}`);
    }
  };

  const uploadMsg = async (file: File) => {
    setUploading(true);
    try {
      const form = new FormData();
      form.append("file", file);
      const url = applicationId
        ? `${apiBase}/api/v1/documents/upload?application_id=${encodeURIComponent(applicationId)}`
        : `${apiBase}/api/v1/documents/upload`;
      const res = await fetch(url, { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.detail || res.statusText);
      setDocIds((ids) => [data.id, ...ids]);
      appendLog(`Uploaded ${data.filename} as document ${data.id}`);
    } catch (e: any) {
      appendLog(`Upload failed: ${e.message}`);
    } finally {
      setUploading(false);
    }
  };

  const triggerOCR = async (docId: string) => {
    try {
      const res = await fetch(`${apiBase}/api/v1/documents/${docId}/ocr`, { method: "POST" });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.detail || res.statusText);
      appendLog(`Queued OCR for ${docId} (task ${data.task_id || "n/a"})`);
    } catch (e: any) {
      appendLog(`Trigger OCR failed: ${e.message}`);
    }
  };

  const processEmailsNow = async () => {
    try {
      const res = await fetch(`${apiBase}/api/v1/email/emails/process-now`, { method: "POST" });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.detail || res.statusText);
      appendLog(`Processed emails now: ${JSON.stringify(data)}`);
    } catch (e: any) {
      appendLog(`Process emails failed: ${e.message}`);
    }
  };

  const startWorkflow = async () => {
    try {
      const url = `${apiBase}/api/v1/task-monitoring/workflows/start?application_id=${encodeURIComponent(applicationId)}&` +
        docIds.map((d) => `document_ids=${encodeURIComponent(d)}`).join("&");
      const res = await fetch(url, { method: "POST" });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.detail || res.statusText);
      appendLog(`Workflow started: ${data.workflow_id}`);
    } catch (e: any) {
      appendLog(`Start workflow failed: ${e.message}`);
    }
  };

  const generateReport = async () => {
    try {
      if (!applicationId) throw new Error("No applicationId");
      const res = await fetch(`${apiBase}/api/v1/reports/applications/${applicationId}/analysis-report`, { method: "POST" });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.detail || res.statusText);
      appendLog(`Report generated: ${data.filename} -> ${data.download_url}`);
    } catch (e: any) {
      appendLog(`Generate report failed: ${e.message}`);
    }
  };

  const loadAnalytics = async () => {
    try {
      const res = await fetch(`${apiBase}/api/v1/analytics/dashboard`);
      const data = await res.json();
      appendLog(`Dashboard metrics: ${JSON.stringify(data)}`);
    } catch (e: any) {
      appendLog(`Analytics failed: ${e.message}`);
    }
  };

  return (
    <div className="space-y-6 p-4">
      <h1 className="text-2xl font-semibold">Demo Orchestrator</h1>

      <div className="grid gap-4 md:grid-cols-2">
        <div className="space-y-3 p-4 border rounded">
          <h2 className="font-medium">1) Application</h2>
          <div className="flex gap-2 items-center">
            <button className="px-3 py-1 border rounded" onClick={createApplication}>Create Application</button>
            <input className="border rounded px-2 py-1 flex-1" placeholder="Application ID" value={applicationId} onChange={(e)=>setApplicationId(e.target.value)} />
          </div>
        </div>

        <div className="space-y-3 p-4 border rounded">
          <h2 className="font-medium">2) Upload .msg / PDFs</h2>
          <input type="file" accept=".msg,.eml,.pdf,.png,.jpg,.jpeg" onChange={(e)=>{ const f=e.target.files?.[0]; if (f) uploadMsg(f); }} disabled={uploading} />
          <div className="text-xs text-gray-600">Last docs: {docIds.slice(0,3).join(", ") || "none"}</div>
        </div>

        <div className="space-y-3 p-4 border rounded">
          <h2 className="font-medium">3) Email Polling</h2>
          <button className="px-3 py-1 border rounded" onClick={processEmailsNow}>Process Emails Now</button>
        </div>

        <div className="space-y-3 p-4 border rounded">
          <h2 className="font-medium">4) OCR / Workflow</h2>
          <div className="flex gap-2">
            <button className="px-3 py-1 border rounded" onClick={()=>{ if(docIds[0]) triggerOCR(docIds[0]); else appendLog("No document to OCR"); }}>Trigger OCR (latest doc)</button>
            <button className="px-3 py-1 border rounded" onClick={startWorkflow} disabled={!applicationId || docIds.length===0}>Start Workflow</button>
          </div>
        </div>

        <div className="space-y-3 p-4 border rounded">
          <h2 className="font-medium">5) Report</h2>
          <button className="px-3 py-1 border rounded" onClick={generateReport} disabled={!applicationId}>Generate Analysis Report</button>
        </div>

        <div className="space-y-3 p-4 border rounded">
          <h2 className="font-medium">6) Analytics</h2>
          <button className="px-3 py-1 border rounded" onClick={loadAnalytics}>Load Dashboard Metrics</button>
        </div>
      </div>

      <div className="p-4 border rounded">
        <h2 className="font-medium mb-2">Activity Log</h2>
        <textarea value={log} readOnly className="w-full h-64 border rounded p-2 font-mono text-xs" />
      </div>
    </div>
  );
}
