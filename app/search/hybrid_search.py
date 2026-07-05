import argparse

from app.db.milvus_client import get_milvus_client
from app.embeddings.embedding_service import EmbeddingService


DEFAULT_COLLECTION_NAME = "industrial_documents_hnsw"
DEFAULT_CANDIDATE_K = 20

OUTPUT_FIELDS = [
    "title",
    "chunk_text",
    "domain",
    "system_name",
    "doc_type",
    "source_file",
    "source_type",
    "chunk_index",
    "page_number",
    "ingested_at",
    "content_hash",
]


def keyword_boost_score(query: str, title: str, chunk_text: str) -> tuple[float, list[str]]:
    query_tokens = {token for token in query.lower().split() if len(token) >= 3}
    searchable_text = f"{title} {chunk_text}".lower()

    boost = 0.0
    matched_keywords = []

    for token in query_tokens:
        if token in searchable_text:
            boost += 0.05
            matched_keywords.append(token)

    return min(boost, 0.30), matched_keywords


def hybrid_search(
    query: str,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    filter_expr: str | None = None,
    top_k: int = 5,
    candidate_k: int = DEFAULT_CANDIDATE_K,
):
    client = get_milvus_client()
    embedder = EmbeddingService()

    query_vector = embedder.embed_text(query)

    search_kwargs = {
        "collection_name": collection_name,
        "data": [query_vector],
        "anns_field": "embedding",
        "limit": candidate_k,
        "output_fields": OUTPUT_FIELDS,
    }

    if filter_expr:
        search_kwargs["filter"] = filter_expr

    results = client.search(**search_kwargs)

    candidates = []

    for hits in results:
        for hit in hits:
            entity = hit["entity"]
            vector_score = hit["distance"]
            boost, matched_keywords = keyword_boost_score(
                query=query,
                title=entity.get("title", ""),
                chunk_text=entity.get("chunk_text", ""),
            )

            candidates.append(
                {
                    "vector_score": vector_score,
                    "keyword_boost": boost,
                    "matched_keywords": matched_keywords,
                    "final_score": vector_score + boost,
                    "entity": entity,
                }
            )

    candidates.sort(key=lambda item: item["final_score"], reverse=True)
    top_results = candidates[:top_k]

    print(f"\nQuery: {query}")
    print(f"Collection: {collection_name}")

    if filter_expr:
        print(f"Filter: {filter_expr}")

    print("=" * 100)

    if not top_results:
        print("No matching documents found.")
        return []

    for rank, item in enumerate(top_results, start=1):
        entity = item["entity"]
        chunk_text = entity.get("chunk_text", "")
        preview = chunk_text if len(chunk_text) <= 220 else f"{chunk_text[:220].rstrip()}..."

        print(f"\nRank: {rank}")
        print(f"Title: {entity.get('title')}")
        print(f"Final Score: {item['final_score']:.4f}")
        print(f"Vector Score: {item['vector_score']:.4f}")
        print(f"Keyword Boost: {item['keyword_boost']:.4f}")
        print(
            "Matched Keywords: "
            + (", ".join(item["matched_keywords"]) if item["matched_keywords"] else "None")
        )
        print(f"Domain: {entity.get('domain')}")
        print(f"System: {entity.get('system_name')}")
        print(f"Doc Type: {entity.get('doc_type')}")
        print(f"Source File: {entity.get('source_file')}")
        print(f"Chunk Index: {entity.get('chunk_index')}")
        print(f"Chunk Text Preview: {preview}")

    return top_results


def main():
    parser = argparse.ArgumentParser(description="Milvus hybrid search CLI")
    parser.add_argument("--query", required=True, help="Natural language search query")
    parser.add_argument(
        "--collection-name",
        default=DEFAULT_COLLECTION_NAME,
        help="Milvus collection name to search",
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
        help="Number of final results to return",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=DEFAULT_CANDIDATE_K,
        help="Number of dense candidates to retrieve before reranking",
    )

    args = parser.parse_args()

    hybrid_search(
        query=args.query,
        collection_name=args.collection_name,
        filter_expr=args.filter,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
    )


if __name__ == "__main__":
    main()
