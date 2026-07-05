from pathlib import Path

from app.ingestion.loaders.base_loader import BaseDocumentLoader


class PdfDocumentLoader(BaseDocumentLoader):
    supported_extensions = {".pdf"}

    def load(self, file_path: Path) -> str:
        raise NotImplementedError("PDF loader will be implemented in a later phase.")