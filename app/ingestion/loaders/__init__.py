from pathlib import Path

from app.ingestion.loaders.txt_loader import TxtDocumentLoader
from app.ingestion.loaders.pdf_loader import PdfDocumentLoader
from app.ingestion.loaders.docx_loader import DocxDocumentLoader
from app.ingestion.loaders.csv_loader import CsvDocumentLoader
from app.ingestion.loaders.log_loader import LogDocumentLoader


LOADERS = [
    TxtDocumentLoader(),
    PdfDocumentLoader(),
    DocxDocumentLoader(),
    CsvDocumentLoader(),
    LogDocumentLoader(),
]


def get_loader_for_file(file_path: Path):
    for loader in LOADERS:
        if loader.supports(file_path):
            return loader

    return None