import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from kumo_label.routers.datasets import router as datasets_router
from kumo_label.routers.embeddings import router as embeddings_router
from kumo_label.routers.jobs import router as jobs_router
from kumo_label.routers.projections import router as projections_router
from kumo_label.routers.training import router as training_router


app = FastAPI(title="Kumo Label API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(datasets_router)
app.include_router(jobs_router)
app.include_router(embeddings_router)
app.include_router(projections_router)
app.include_router(training_router)


@app.get("/api/health")
def health():
    return {"status": "ok"}


class SPAStaticFiles(StaticFiles):
    """Static files with index.html fallback, so client-side routes
    like /explorer survive a hard refresh. Unmatched /api paths keep
    their 404 instead of receiving HTML."""

    async def get_response(self, path: str, scope):
        try:
            response = await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code == 404 and not path.startswith("api/"):
                return await super().get_response("index.html", scope)
            raise
        if response.status_code == 404 and not path.startswith("api/"):
            return await super().get_response("index.html", scope)
        return response


# In Docker/production the built frontend is served from the same origin;
# API routes above take precedence over this catch-all mount.
_static_dir = os.environ.get("KUMO_STATIC_DIR")
if _static_dir and Path(_static_dir).is_dir():
    app.mount("/", SPAStaticFiles(directory=_static_dir, html=True), name="frontend")
