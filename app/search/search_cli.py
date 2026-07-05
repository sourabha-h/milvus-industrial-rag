import argparse

from app.db.milvus_client import get_milvus_client
from app.embeddings.embedding_service import EmbeddingService


COLLECTION_NAME = "industrial_documents"

OUTPUT_FIELDS = [
    "doc_id",
    "source_file",
    "source_type",
    "title",
    "chunk_text",
    "domain",
    "system_name",
    "doc_type",
    "chunk_index",
    "page_number",
    "created_date",
    "ingested_at",
    "content_hash",
]


def search(query: str, filter_expr: str | None = None, top_k: int = 5):
    client = get_milvus_client()
    embedder = EmbeddingService()

    query_vector = embedder.embed_text(query)

    search_kwargs = {
        "collection_name": COLLECTION_NAME,
        "data": [query_vector],
        "anns_field": "embedding",
        "limit": top_k,
        "output_fields": OUTPUT_FIELDS,
    }

    if filter_expr:
        search_kwargs["filter"] = filter_expr

    results = client.search(**search_kwargs)

    print(f"\nQuery: {query}")

    if filter_expr:
        print(f"Filter: {filter_expr}")

    print("=" * 100)

    for hits in results:
        if not hits:
            print("No matching documents found.")
            return

        for rank, hit in enumerate(hits, start=1):
            entity = hit["entity"]

            print(f"\nRank: {rank}")
            print(f"Score: {hit['distance']}")
            print(f"Title: {entity.get('title')}")
            print(f"Doc ID: {entity.get('doc_id')}")
            print(f"Domain: {entity.get('domain')}")
            print(f"System: {entity.get('system_name')}")
            print(f"Doc Type: {entity.get('doc_type')}")
            print(f"Source File: {entity.get('source_file')}")
            print(f"Source Type: {entity.get('source_type')}")
            print(f"Chunk Index: {entity.get('chunk_index')}")
            print(f"Page Number: {entity.get('page_number')}")
            print(f"Ingested At: {entity.get('ingested_at')}")
            print(f"Content Hash: {entity.get('content_hash')}")
            print(f"Text: {entity.get('chunk_text')}")


def main():
    parser = argparse.ArgumentParser(
        description="Milvus Industrial RAG semantic search CLI"
    )

    parser.add_argument(
        "--query",
        required=True,
        help="Natural language search query",
    )

    parser.add_argument(
        "--filter",
        default=None,
        help='Optional Milvus filter expression, example: domain == "telecom"',
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of search results to return",
    )

    args = parser.parse_args()

    search(
        query=args.query,
        filter_expr=args.filter,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()