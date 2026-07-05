from pathlib import Path
from abc import ABC, abstractmethod


class BaseDocumentLoader(ABC):
    supported_extensions: set[str] = set()

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.supported_extensions

    @abstractmethod
    def load(self, file_path: Path) -> str:
        """Extract text from a source document."""
        raise NotImplementedError