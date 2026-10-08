class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = "ApiError";
  }
}

// Assumes body, if present, is always JSON (set by api.post via JSON.stringify).
async function fetcher<T>(path: string, options?: RequestInit): Promise<T> {
  const headers: Record<string, string> = {};
  if (options?.body) {
    headers["Content-Type"] = "application/json";
  }
  const response = await fetch(`/api${path}`, {
    headers,
    ...options,
  });

  if (!response.ok) {
    let message = `API error ${response.status}: ${response.statusText}`;
    try {
      const body = await response.json();
      if (body?.detail) message = body.detail;
    } catch {
      // ignore parse errors, use default message
    }
    throw new ApiError(response.status, message);
  }

  return response.json() as Promise<T>;
}

async function fetchBinary(path: string): Promise<ArrayBuffer> {
  const response = await fetch(`/api${path}`);
  if (!response.ok) {
    throw new ApiError(response.status, `API error ${response.status}: ${response.statusText}`);
  }
  return response.arrayBuffer();
}

export const api = {
  get: <T>(path: string) => fetcher<T>(path),
  post: <T>(path: string, body: unknown) =>
    fetcher<T>(path, { method: "POST", body: JSON.stringify(body) }),
  put: <T>(path: string, body: unknown) =>
    fetcher<T>(path, { method: "PUT", body: JSON.stringify(body) }),
  patch: <T>(path: string, body: unknown) =>
    fetcher<T>(path, { method: "PATCH", body: JSON.stringify(body) }),
  delete: <T>(path: string) => fetcher<T>(path, { method: "DELETE" }),
  getBinary: (path: string) => fetchBinary(path),
};

export { ApiError };
