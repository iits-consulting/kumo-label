import { createQuery } from "@tanstack/svelte-query";
import { api } from "@/lib/api";

export interface JobStatus {
  id: string;
  type: "embedding" | "projection";
  status: "queued" | "running" | "done" | "failed" | "cancelled";
  message: string | null;
  progress: number;
  completed: number;
  total: number;
  error: string | null;
}

// Accessor args (not plain values) so the query re-keys reactively when the
// caller's $state changes — svelte-query v6 createQuery takes an options thunk.
export function createJobPoller(jobId: () => string | null, dbPath: () => string | null) {
  return createQuery<JobStatus>(() => ({
    queryKey: ["job", jobId(), dbPath()],
    queryFn: () => api.get<JobStatus>(`/jobs/${jobId()}?db_path=${encodeURIComponent(dbPath()!)}`),
    enabled: !!jobId() && !!dbPath(),
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status === "done" || status === "failed" || status === "cancelled" ? false : 2000;
    },
  }));
}
