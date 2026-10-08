<script lang="ts">
  // Test-only harness: puts a QueryClient on context, creates the poller
  // during component init, and hands the reactive result back to the test.
  import { setQueryClientContext, type QueryClient } from "@tanstack/svelte-query";
  import { createJobPoller } from "@/state/jobPoller";

  let {
    client,
    jobId,
    dbPath,
    expose,
  }: {
    client: QueryClient;
    jobId: string | null;
    dbPath: string | null;
    expose: (query: ReturnType<typeof createJobPoller>) => void;
  } = $props();

  // svelte-ignore state_referenced_locally -- init-time capture is intentional here
  setQueryClientContext(client);
  // svelte-ignore state_referenced_locally -- init-time capture is intentional here
  expose(createJobPoller(() => jobId, () => dbPath));
</script>
