import re


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_doc_id(source_file_name: str) -> str:
    return source_file_name.lower().replace(" ", "_")