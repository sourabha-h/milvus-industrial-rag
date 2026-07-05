from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import argparse

from app.db.milvus_client import get_milvus_client
from app.embeddings.embedding_service import EmbeddingService
from app.ingestion.chunker import chunk_text
from app.ingestion.loaders import get_loader_for_file
from app.ingestion.metadata_extractor import infer_metadata
from app.ingestion.normalizer import clean_text, build_doc_id


COLLECTION_NAME = "industrial_documents"
RAW_DATA_DIR = Path("data/raw")


def find_candidate_files(raw_data_dir: Path) -> list[Path]:
    files = []

    for file_path in raw_data_dir.rglob("*"):
        if not file_path.is_file():
            continue

        loader = get_loader_for_file(file_path)

        if loader is not None:
            files.append(file_path)

    return files


def normalize_source_file(file_path: Path) -> str:
    return file_path.as_posix()


def build_row_id(source_file: str, chunk_index: int, content_hash: str) -> int:
    digest = sha256(f"{source_file}|{chunk_index}|{content_hash}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def build_content_hash(cleaned_chunk_text: str) -> str:
    return sha256(cleaned_chunk_text.encode("utf-8")).hexdigest()


def chunk_exists(client, collection_name: str, content_hash: str) -> bool:
    results = client.query(
        collection_name=collection_name,
        filter=f'content_hash == "{content_hash}"',
        output_fields=["content_hash"],
    )

    return bool(results)


def ingest_documents(input_dir: Path = RAW_DATA_DIR, collection_name: str = COLLECTION_NAME):
    client = get_milvus_client()
    embedder = EmbeddingService()

    files = find_candidate_files(input_dir)

    if not files:
        print(f"No supported files found under {input_dir}")
        return

    rows = []

    for file_path in files:
        loader = get_loader_for_file(file_path)

        if loader is None:
            print(f"Skipping unsupported file: {file_path}")
            continue

        try:
            print(f"Processing: {file_path}")
            raw_text = loader.load(file_path)

        except NotImplementedError as exc:
            print(f"Skipping placeholder loader for {file_path}: {exc}")
            continue

        except Exception as exc:
            print(f"Failed to load {file_path}: {exc}")
            continue

        metadata = infer_metadata(file_path, raw_text)

        cleaned_text = clean_text(raw_text)

        if not cleaned_text:
            print(f"Skipping empty document: {file_path}")
            continue

        chunks = chunk_text(cleaned_text)

        if not chunks:
            print(f"No chunks generated for: {file_path}")
            continue

        embeddings = embedder.embed_texts(chunks)
        doc_id = build_doc_id(file_path.stem)
        source_file = normalize_source_file(file_path)
        source_type = file_path.suffix.lstrip(".").lower()
        ingested_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

        for chunk_index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            cleaned_chunk = clean_text(chunk)
            content_hash = build_content_hash(cleaned_chunk)

            if chunk_exists(client, collection_name, content_hash):
                print(
                    f"Skipping duplicate chunk from {source_file} "
                    f"(chunk_index={chunk_index}, content_hash={content_hash})"
                )
                continue

            row = {
                "id": build_row_id(source_file, chunk_index, content_hash),
                "doc_id": doc_id,
                "title": metadata["title"],
                "chunk_text": chunk,
                "domain": metadata["domain"],
                "system_name": metadata["system_name"],
                "doc_type": metadata["doc_type"],
                "created_date": datetime.now().strftime("%Y-%m-%d"),
                "source_file": source_file,
                "source_type": source_type,
                "chunk_index": chunk_index,
                "page_number": -1,
                "ingested_at": ingested_at,
                "content_hash": content_hash,
                "embedding": embedding,
            }

            rows.append(row)

    if not rows:
        print("No new rows to insert.")
        return

    result = client.insert(
        collection_name=collection_name,
        data=rows,
    )

    print(f"Inserted {len(rows)} chunks from {len(files)} candidate files.")
    print(result)


def main():
    parser = argparse.ArgumentParser(description="Ingest supported documents into Milvus")
    parser.add_argument(
        "--input-dir",
        default=str(RAW_DATA_DIR),
        help="Input directory containing supported documents",
    )

    parser.add_argument(
        "--collection-name",
        default=COLLECTION_NAME,
        help="Milvus collection name to ingest into",
    )

    args = parser.parse_args()
    ingest_documents(
        input_dir=Path(args.input_dir),
        collection_name=args.collection_name,
    )


if __name__ == "__main__":
    main()
