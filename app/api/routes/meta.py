from fastapi import APIRouter
from fastapi.responses import FileResponse

from app.config import PROJECT_ROOT

router = APIRouter()
UI_PATH = PROJECT_ROOT / "frontend" / "index.html"


@router.get("/health", response_model=dict[str, str])
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ui", include_in_schema=False)
async def test_ui() -> FileResponse:
    return FileResponse(UI_PATH)
