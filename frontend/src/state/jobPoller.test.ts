import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, waitFor } from "@testing-library/svelte";
import { QueryClient } from "@tanstack/svelte-query";
import PollerHarness from "@/test/PollerHarness.svelte";
import type { createJobPoller } from "./jobPoller";

type PollerResult = ReturnType<typeof createJobPoller>;

function renderPoller(jobId: string | null, dbPath: string | null): PollerResult {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  let result: PollerResult;
  render(PollerHarness, {
    props: { client, jobId, dbPath, expose: (q: PollerResult) => (result = q) },
  });
  return result!;
}

function mockFetch(body: unknown, status = 200) {
  return vi.fn().mockResolvedValue({
    ok: status >= 200 && status < 300,
    status,
    statusText: "OK",
    json: () => Promise.resolve(body),
  });
}

beforeEach(() => vi.restoreAllMocks());

describe("createJobPoller", () => {
  it("returns undefined when jobId is null", () => {
    const query = renderPoller(null, "/a/kumo.db");
    expect(query.data).toBeUndefined();
  });

  it("fetches job status when jobId and dbPath are set", async () => {
    const job = { id: "abc", status: "done", progress: 1, completed: 10, total: 10, error: null, message: null, type: "embedding" };
    vi.stubGlobal("fetch", mockFetch(job));

    const query = renderPoller("abc", "/data/kumo.db");

    await waitFor(() => expect(query.data).toBeDefined());
    expect(query.data?.status).toBe("done");
  });

  it("does not fetch when dbPath is null", () => {
    const fetchMock = mockFetch({});
    vi.stubGlobal("fetch", fetchMock);
    renderPoller("abc", null);
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
