from pathlib import Path

from app.ingestion.loaders.base_loader import BaseDocumentLoader


class TxtDocumentLoader(BaseDocumentLoader):
    supported_extensions = {".txt"}

    def load(self, file_path: Path) -> str:
        return file_path.read_text(encoding="utf-8", errors="ignore")