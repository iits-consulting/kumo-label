import { describe, it, expect, vi, beforeEach } from "vitest";
import { api, ApiError } from "./api";

function mockFetch(status: number, body: unknown) {
  return vi.fn().mockResolvedValue({
    ok: status >= 200 && status < 300,
    status,
    statusText: status === 200 ? "OK" : "Error",
    json: () => Promise.resolve(body),
  });
}

beforeEach(() => {
  vi.restoreAllMocks();
});

describe("api.get", () => {
  it("returns parsed JSON on success", async () => {
    vi.stubGlobal("fetch", mockFetch(200, { status: "ok" }));
    const result = await api.get<{ status: string }>("/health");
    expect(result).toEqual({ status: "ok" });
  });

  it("prepends /api to the path", async () => {
    const fetchMock = mockFetch(200, {});
    vi.stubGlobal("fetch", fetchMock);
    await api.get("/health");
    expect(fetchMock).toHaveBeenCalledWith("/api/health", expect.any(Object));
  });

  it("does not set Content-Type on GET", async () => {
    const fetchMock = mockFetch(200, {});
    vi.stubGlobal("fetch", fetchMock);
    await api.get("/health");
    const headers = fetchMock.mock.calls[0][1].headers;
    expect(headers["Content-Type"]).toBeUndefined();
  });

  it("throws ApiError on non-2xx response", async () => {
    vi.stubGlobal("fetch", mockFetch(404, {}));
    await expect(api.get("/missing")).rejects.toBeInstanceOf(ApiError);
  });

  it("ApiError carries the status code", async () => {
    vi.stubGlobal("fetch", mockFetch(500, {}));
    try {
      await api.get("/fail");
    } catch (e) {
      expect(e).toBeInstanceOf(ApiError);
      expect((e as ApiError).status).toBe(500);
    }
  });
});

describe("api.post", () => {
  it("sends JSON body and returns parsed response", async () => {
    const fetchMock = mockFetch(200, { id: 1 });
    vi.stubGlobal("fetch", fetchMock);
    const result = await api.post<{ id: number }>("/items", { name: "test" });
    expect(result).toEqual({ id: 1 });
    const call = fetchMock.mock.calls[0][1];
    expect(call.method).toBe("POST");
    expect(call.body).toBe(JSON.stringify({ name: "test" }));
    expect(call.headers["Content-Type"]).toBe("application/json");
  });

  it("throws ApiError on non-2xx response", async () => {
    vi.stubGlobal("fetch", mockFetch(400, {}));
    await expect(api.post("/items", {})).rejects.toBeInstanceOf(ApiError);
  });
});

describe("api.patch", () => {
  it("prepends /api to the path", async () => {
    const fetchMock = mockFetch(200, {});
    vi.stubGlobal("fetch", fetchMock);
    await api.patch("/datasets/images/labels", {});
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/datasets/images/labels",
      expect.any(Object)
    );
  });

  it("sends PATCH method, JSON body, and Content-Type header", async () => {
    const fetchMock = mockFetch(200, { updated: 2 });
    vi.stubGlobal("fetch", fetchMock);
    const body = { db_path: "/a/kumo.db", ids: [1, 2], label: "cat" };
    const result = await api.patch<{ updated: number }>(
      "/datasets/images/labels",
      body
    );
    expect(result).toEqual({ updated: 2 });
    const call = fetchMock.mock.calls[0][1];
    expect(call.method).toBe("PATCH");
    expect(call.body).toBe(JSON.stringify(body));
    expect(call.headers["Content-Type"]).toBe("application/json");
  });

  it("throws ApiError on non-2xx response", async () => {
    vi.stubGlobal("fetch", mockFetch(400, {}));
    await expect(
      api.patch("/datasets/images/labels", {})
    ).rejects.toBeInstanceOf(ApiError);
  });
});
