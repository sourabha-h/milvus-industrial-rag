from pathlib import Path

from app.ingestion.loaders.base_loader import BaseDocumentLoader


class LogDocumentLoader(BaseDocumentLoader):
    supported_extensions = {".log"}

    def load(self, file_path: Path) -> str:
        raise NotImplementedError("LOG loader will be implemented in a later phase.")