import { createQuery } from "@tanstack/svelte-query";
import { api } from "@/lib/api";
import type { TrainingRun } from "@/lib/types";

// Accessor args (not plain values) so the query re-keys reactively when the
// caller's $state changes — same shape as createJobPoller. No side effects
// here: completion behavior lives in the consumer (TrainingDrawer), keyed on
// status only.
export function createTrainingPoller(runId: () => string | null, dbPath: () => string | null) {
  return createQuery<TrainingRun>(() => ({
    queryKey: ["training-run", runId(), dbPath()],
    queryFn: () =>
      api.get<TrainingRun>(`/training/runs/${runId()}?db_path=${encodeURIComponent(dbPath()!)}`),
    enabled: !!runId() && !!dbPath(),
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status === "done" || status === "failed" || status === "stopped" ? false : 2000;
    },
  }));
}
