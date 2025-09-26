"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "../../lib/utils";
import { useEffect, useState } from "react";

const NavLink = ({ href, label }: { href: string; label: string }) => {
  const pathname = usePathname();
  const active = pathname === href;
  return (
    <Link
      href={href}
      className={cn(
        "px-3 py-2 rounded-md text-sm font-medium",
        active
          ? "bg-primary text-primary-foreground"
          : "text-foreground hover:bg-muted hover:text-foreground"
      )}
    >
      {label}
    </Link>
  );
};

export default function Navbar() {
  const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8005";
  const [status, setStatus] = useState<{
    polling_task_running: boolean;
    agent_running: boolean;
    last_run_at?: string | null;
    last_processed?: number;
    last_successful?: number;
    last_errors?: number;
    total_processed?: number;
    total_successful?: number;
    total_errors?: number;
  }>({ polling_task_running: false, agent_running: false });
  const [open, setOpen] = useState(false);

  const refreshStatus = async () => {
    try {
      const res = await fetch(`${apiBase}/api/v1/email/emails/polling/status`, {
        cache: "no-store",
      });
      if (res.ok) {
        const data = await res.json();
        setStatus(data);
      }
    } catch (e) {
      // ignore
    }
  };

  useEffect(() => {
    // initial and interval refresh
    refreshStatus();
    const id = setInterval(refreshStatus, 5000);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const togglePolling = async () => {
    const endpoint = status.polling_task_running ? "disable" : "enable";
    try {
      await fetch(`${apiBase}/api/v1/email/emails/polling/${endpoint}`, {
        method: "POST",
      });
      await refreshStatus();
    } catch (e) {
      // ignore
    }
  };

  const processNow = async () => {
    try {
      const res = await fetch(`${apiBase}/api/v1/email/emails/process-now`, {
        method: "POST",
      });
      if (res.ok) {
        await refreshStatus();
      }
    } catch (e) {
      // ignore
    }
  };

  return (
    <header className="border-b bg-background">
      <div className="container flex h-14 items-center justify-between">
        <Link href="/" className="flex items-center gap-2">
          <img src="/images/kenya-re.png" alt="Kenya Re logo" className="h-8" />
        </Link>
        <nav className="flex items-center gap-4">
          <NavLink href="/" label="Home" />
          <NavLink href="/dashboard" label="Dashboard" />
          <NavLink href="/upload" label="Upload" />
          <NavLink href="/documents" label="Documents" />
          <NavLink href="/applications" label="Applications" />
          <NavLink href="/demo" label="Demo" />
          <NavLink href="/login" label="Login" />
          <div className="relative">
            <button
              onClick={() => setOpen((v) => !v)}
              className={cn(
                "flex items-center gap-2 text-xs px-2 py-1 rounded border hover:bg-muted",
              )}
              aria-haspopup="true"
              aria-expanded={open}
              title="Email polling status"
            >
              <span
                className={cn(
                  "inline-block h-2 w-2 rounded-full",
                  status.polling_task_running ? "bg-green-500" : "bg-gray-400"
                )}
                aria-label={status.polling_task_running ? "Polling enabled" : "Polling disabled"}
              />
              {status.polling_task_running ? "Polling: On" : "Polling: Off"}
            </button>
            {open && (
              <div
                className="absolute right-0 mt-2 w-64 rounded-md border bg-popover p-3 shadow-md z-50"
                role="dialog"
                aria-label="Email polling details"
              >
                <div className="mb-2 text-sm">
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">Agent</span>
                    <span
                      className={cn(
                        "ml-2 inline-flex items-center gap-1 text-xs",
                        status.agent_running ? "text-green-600" : "text-muted-foreground"
                      )}
                    >
                      <span
                        className={cn(
                          "inline-block h-2 w-2 rounded-full",
                          status.agent_running ? "bg-green-500" : "bg-gray-400"
                        )}
                        aria-hidden
                      />
                      {status.agent_running ? "Running" : "Idle"}
                    </span>
                  </div>
                  <div className="mt-1">
                    <div className="text-muted-foreground">Last run</div>
                    <div className="font-mono text-xs break-all">
                      {status.last_run_at ? new Date(status.last_run_at).toLocaleString() : "—"}
                    </div>
                  </div>
                </div>
                <div className="grid grid-cols-3 gap-2 text-center">
                  <div className="rounded bg-muted p-2">
                    <div className="text-xs text-muted-foreground">Processed</div>
                    <div className="font-semibold">{status.last_processed ?? 0}</div>
                  </div>
                  <div className="rounded bg-muted p-2">
                    <div className="text-xs text-green-700">OK</div>
                    <div className="font-semibold text-green-700">{status.last_successful ?? 0}</div>
                  </div>
                  <div className="rounded bg-muted p-2">
                    <div className="text-xs text-red-700">Errors</div>
                    <div className="font-semibold text-red-700">{status.last_errors ?? 0}</div>
                  </div>
                </div>
                <div className="mt-2 grid grid-cols-3 gap-2 text-center">
                  <div className="rounded bg-muted p-2">
                    <div className="text-[10px] text-muted-foreground">Total Proc.</div>
                    <div className="font-semibold">{status.total_processed ?? 0}</div>
                  </div>
                  <div className="rounded bg-muted p-2">
                    <div className="text-[10px] text-green-700">Total OK</div>
                    <div className="font-semibold text-green-700">{status.total_successful ?? 0}</div>
                  </div>
                  <div className="rounded bg-muted p-2">
                    <div className="text-[10px] text-red-700">Total Err</div>
                    <div className="font-semibold text-red-700">{status.total_errors ?? 0}</div>
                  </div>
                </div>
                <div className="mt-3 flex items-center justify-between gap-2">
                  <button
                    onClick={togglePolling}
                    className="text-xs px-2 py-1 rounded border hover:bg-muted"
                  >
                    {status.polling_task_running ? "Disable" : "Enable"}
                  </button>
                  <button
                    onClick={processNow}
                    className="text-xs px-2 py-1 rounded bg-primary text-primary-foreground hover:opacity-90"
                  >
                    Process Now
                  </button>
                </div>
              </div>
            )}
          </div>
        </nav>
      </div>
    </header>
  );
}
