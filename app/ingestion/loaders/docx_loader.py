from pathlib import Path

from app.ingestion.loaders.base_loader import BaseDocumentLoader


class DocxDocumentLoader(BaseDocumentLoader):
    supported_extensions = {".docx"}

    def load(self, file_path: Path) -> str:
        raise NotImplementedError("DOCX loader will be implemented in a later phase.")