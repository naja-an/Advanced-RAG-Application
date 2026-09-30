"""On-disk storage for uploaded files (kept only while they are being parsed)."""

import re
import shutil
from pathlib import Path
from uuid import UUID, uuid4


class UploadStorage:
    def __init__(self, root: Path):
        self._root = Path(root)

    def session_dir(self, session_id: UUID) -> Path:
        path = self._root / str(session_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save(self, session_id: UUID, filename: str, content: bytes) -> Path:
        safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", Path(filename).name)
        path = self.session_dir(session_id) / f"{uuid4()}-{safe_name}"
        path.write_bytes(content)
        return path

    def remove(self, session_id: UUID) -> None:
        shutil.rmtree(self._root / str(session_id), ignore_errors=True)
