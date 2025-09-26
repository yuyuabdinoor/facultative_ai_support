"use client";
import { useMemo, useRef, useState } from "react";

type FileItem = {
  file: File;
  id: string;
  progress: number;
  status: "pending" | "uploading" | "done" | "error";
  message?: string;
  docId?: string;
};

const ACCEPTED_MIME = [
  "application/pdf",
  "application/vnd.ms-outlook",
  "application/vnd.ms-office",
  "message/rfc822",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  "application/vnd.ms-excel",
  "application/vnd.openxmlformats-officedocument.presentationml.presentation",
  "image/png",
  "image/jpeg",
];

const ACCEPTED_EXT = [".pdf", ".msg", ".docx", ".xlsx", ".xls", ".pptx", ".png", ".jpg", ".jpeg"];

export default function UploadPage() {
  const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8005";
  const [items, setItems] = useState<FileItem[]>([]);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const totalProgress = useMemo(() => {
    const total = items.length || 1;
    const sum = items.reduce((acc, it) => acc + it.progress, 0);
    return Math.round(sum / total);
  }, [items]);

  const validateFile = (f: File): string | null => {
    const ext = `.${f.name.split(".").pop()?.toLowerCase()}`;
    if (!ACCEPTED_EXT.includes(ext) && !ACCEPTED_MIME.includes(f.type)) {
      return `Unsupported file type: ${ext || f.type || "unknown"}`;
    }
    const maxSize = 100 * 1024 * 1024; // 100MB
    if (f.size > maxSize) return "File too large (max 100MB)";
    return null;
  };

  const addFiles = (fileList: FileList | null) => {
    if (!fileList) return;
    const next: FileItem[] = [];
    Array.from(fileList).forEach((f) => {
      const err = validateFile(f);
      const id = `${Date.now()}_${Math.random().toString(36).slice(2)}`;
      next.push({
        file: f,
        id,
        progress: 0,
        status: err ? "error" : "pending",
        message: err || undefined,
      });
    });
    setItems((prev) => [...prev, ...next]);
  };

  const onDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragOver(false);
    addFiles(e.dataTransfer.files);
  };

  const onDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragOver(true);
  };

  const onDragLeave = () => setDragOver(false);

  const pickFiles = () => inputRef.current?.click();

  const uploadItem = (item: FileItem) =>
    new Promise<FileItem>((resolve) => {
      if (item.status === "error") return resolve(item);
      const xhr = new XMLHttpRequest();
      xhr.open("POST", `${apiBase}/api/v1/documents/upload`);
      xhr.upload.onprogress = (evt) => {
        if (evt.lengthComputable) {
          const pct = Math.round((evt.loaded / evt.total) * 100);
          setItems((prev) => prev.map((it) => (it.id === item.id ? { ...it, progress: pct } : it)));
        }
      };
      xhr.onreadystatechange = () => {
        if (xhr.readyState === 4) {
          if (xhr.status >= 200 && xhr.status < 300) {
            try {
              const data = JSON.parse(xhr.responseText);
              resolve({ ...item, status: "done", progress: 100, docId: data?.id, message: "Uploaded" });
            } catch {
              resolve({ ...item, status: "done", progress: 100, message: "Uploaded" });
            }
          } else {
            resolve({ ...item, status: "error", message: `Failed (${xhr.status})` });
          }
        }
      };
      const form = new FormData();
      form.append("file", item.file);
      xhr.send(form);
      setItems((prev) => prev.map((it) => (it.id === item.id ? { ...it, status: "uploading", progress: 1 } : it)));
    });

  const uploadAll = async () => {
    const toUpload = items.filter((i) => i.status === "pending");
    const updated: FileItem[] = [];
    for (const it of toUpload) {
      // sequential to better reflect progress
      const res = await uploadItem(it);
      updated.push(res);
    }
    if (updated.length) {
      setItems((prev) => prev.map((p) => updated.find((u) => u.id === p.id) || p));
    }
  };

  const removeItem = (id: string) => setItems((prev) => prev.filter((i) => i.id !== id));

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold">Upload Documents</h1>
      <p className="text-muted-foreground">
        Upload PDFs, Excel files, images, or Outlook .msg emails. Files will be validated and stored for processing.
      </p>

      <div
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        className={
          "border-2 border-dashed rounded-md p-8 text-center transition " +
          (dragOver ? "bg-muted/70 border-primary" : "hover:bg-muted/50")
        }
      >
        <p className="mb-3">Drag & drop files here</p>
        <p className="mb-3">or</p>
        <button onClick={pickFiles} className="px-3 py-2 rounded border">Choose Files</button>
        <input
          ref={inputRef}
          type="file"
          className="hidden"
          multiple
          accept={ACCEPTED_EXT.join(",")}
          onChange={(e) => addFiles(e.target.files)}
        />
        <div className="mt-4 text-xs text-muted-foreground">
          Accepted: {ACCEPTED_EXT.join(", ")}
        </div>
      </div>

      {items.length > 0 && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="text-sm text-muted-foreground">Batch Progress</div>
            <div className="text-sm font-medium">{totalProgress}%</div>
          </div>
          <div className="h-2 w-full rounded bg-muted">
            <div className="h-2 rounded bg-primary" style={{ width: `${totalProgress}%` }} />
          </div>

          <ul className="divide-y rounded border">
            {items.map((it) => (
              <li key={it.id} className="p-3 flex items-center justify-between gap-4">
                <div className="min-w-0">
                  <div className="truncate font-medium">{it.file.name}</div>
                  <div className="text-xs text-muted-foreground">
                    {(it.file.size / 1024).toFixed(0)} KB • {it.file.type || it.file.name.split(".").pop()?.toUpperCase()}
                  </div>
                </div>
                <div className="flex-1 px-4">
                  <div className="h-2 w-full rounded bg-muted">
                    <div
                      className={"h-2 rounded " + (it.status === "error" ? "bg-red-600" : "bg-green-600")}
                      style={{ width: `${it.progress}%` }}
                    />
                  </div>
                  <div className="text-right text-xs mt-1">
                    {it.status === "pending" && "Pending"}
                    {it.status === "uploading" && `Uploading ${it.progress}%`}
                    {it.status === "done" && "Uploaded"}
                    {it.status === "error" && (<span className="text-red-700">{it.message || "Error"}</span>)}
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  {it.docId && (
                    <code className="text-xs bg-muted px-2 py-1 rounded" title="Document ID">{it.docId}</code>
                  )}
                  <button
                    onClick={() => removeItem(it.id)}
                    className="text-xs px-2 py-1 rounded border hover:bg-muted"
                    aria-label={`Remove ${it.file.name}`}
                  >
                    Remove
                  </button>
                </div>
              </li>
            ))}
          </ul>

          <div className="flex items-center gap-3">
            <button
              onClick={uploadAll}
              className="px-4 py-2 rounded bg-primary text-primary-foreground disabled:opacity-50"
              disabled={items.every((i) => i.status !== "pending")}
            >
              Upload All
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
