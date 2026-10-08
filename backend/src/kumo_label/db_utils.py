from pathlib import Path
from fastapi import HTTPException


def validate_db_path(db_path: str) -> Path:
    """Validate and return the resolved db path, or raise HTTP 400."""
    p = Path(db_path)
    if not p.is_absolute():
        raise HTTPException(status_code=400, detail="db_path must be an absolute path")
    if not p.exists():
        raise HTTPException(status_code=400, detail="db_path does not exist")
    if p.name != "kumo.db":
        raise HTTPException(status_code=400, detail="db_path must point to a kumo.db file")
    return p
