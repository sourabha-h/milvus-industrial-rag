from pathlib import Path

from app.ingestion.loaders.base_loader import BaseDocumentLoader


class CsvDocumentLoader(BaseDocumentLoader):
    supported_extensions = {".csv"}

    def load(self, file_path: Path) -> str:
        raise NotImplementedError("CSV loader will be implemented in a later phase.")