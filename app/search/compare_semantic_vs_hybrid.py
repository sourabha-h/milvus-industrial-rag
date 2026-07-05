import argparse

from app.db.milvus_client import get_milvus_client
from app.embeddings.embedding_service import EmbeddingService
from app.search.hybrid_search import OUTPUT_FIELDS, keyword_boost_score


SEMANTIC_COLLECTION_NAME = "industrial_documents"
HYBRID_COLLECTION_NAME = "industrial_documents_hnsw"
DEFAULT_TOP_K = 5
DEFAULT_CANDIDATE_K = 20


def run_semantic_search(query: str, collection_name: str, top_k: int):
    client = get_milvus_client()
    embedder = EmbeddingService()
    query_vector = embedder.embed_text(query)

    results = client.search(
        collection_name=collection_name,
        data=[query_vector],
        anns_field="embedding",
        limit=top_k,
        output_fields=[
            "title",
            "chunk_text",
            "domain",
            "system_name",
            "doc_type",
            "source_file",
            "chunk_index",
        ],
    )

    rows = []

    for hits in results:
        for hit in hits:
            entity = hit["entity"]
            rows.append(
                {
                    "score": hit["distance"],
                    "title": entity.get("title", ""),
                    "domain": entity.get("domain", ""),
                    "system_name": entity.get("system_name", ""),
                    "doc_type": entity.get("doc_type", ""),
                    "source_file": entity.get("source_file", ""),
                    "chunk_index": entity.get("chunk_index", ""),
                    "chunk_text": entity.get("chunk_text", ""),
                }
            )

    return rows


def run_hybrid_search(
    query: str,
    collection_name: str,
    top_k: int,
    candidate_k: int,
    filter_expr: str | None = None,
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
                    "final_score": vector_score + boost,
                    "vector_score": vector_score,
                    "keyword_boost": boost,
                    "matched_keywords": matched_keywords,
                    "title": entity.get("title", ""),
                    "source_file": entity.get("source_file", ""),
                    "chunk_index": entity.get("chunk_index", ""),
                }
            )

    candidates.sort(key=lambda item: item["final_score"], reverse=True)
    return candidates[:top_k]


def result_key(row: dict) -> tuple[str, str]:
    return (str(row.get("source_file", "")), str(row.get("chunk_index", "")))


def compare_semantic_vs_hybrid(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    semantic_collection_name: str = SEMANTIC_COLLECTION_NAME,
    hybrid_collection_name: str = HYBRID_COLLECTION_NAME,
    filter_expr: str | None = None,
):
    semantic_rows = run_semantic_search(query, semantic_collection_name, top_k)
    hybrid_rows = run_hybrid_search(
        query,
        hybrid_collection_name,
        top_k,
        candidate_k,
        filter_expr=filter_expr,
    )

    print(f"\nQuery: {query}")
    print(f"Semantic Collection: {semantic_collection_name}")
    print(f"Hybrid Collection: {hybrid_collection_name}")
    if filter_expr:
        print(f"Filter: {filter_expr}")
    print("=" * 100)

    semantic_rank_map = {result_key(row): index + 1 for index, row in enumerate(semantic_rows)}

    print("Semantic Search")
    print("-" * 100)
    for rank, row in enumerate(semantic_rows[:top_k], start=1):
        print(f"Rank: {rank}")
        print(f"Title: {row.get('title')}")
        print(f"Score: {row.get('score', 0.0):.4f}")
        print(f"Source File: {row.get('source_file')}")
        print(f"Chunk Index: {row.get('chunk_index')}")
        print("-" * 100)

    print("Hybrid Search")
    print("-" * 100)
    for rank, row in enumerate(hybrid_rows[:top_k], start=1):
        semantic_rank = semantic_rank_map.get(result_key(row))
        semantic_rank_text = str(semantic_rank) if semantic_rank is not None else "not in semantic top-k"
        print(f"Rank: {rank}")
        print(f"Semantic Rank: {semantic_rank_text}")
        print(f"Title: {row.get('title')}")
        print(f"Final Score: {row.get('final_score', 0.0):.4f}")
        print(f"Vector Score: {row.get('vector_score', 0.0):.4f}")
        print(f"Keyword Boost: {row.get('keyword_boost', 0.0):.4f}")
        print(
            "Matched Keywords: "
            + (", ".join(row["matched_keywords"]) if row.get("matched_keywords") else "None")
        )
        print(f"Source File: {row.get('source_file')}")
        print(f"Chunk Index: {row.get('chunk_index')}")
        print("-" * 100)


def main():
    parser = argparse.ArgumentParser(description="Compare semantic and hybrid search results")
    parser.add_argument("--query", required=True, help="Natural language search query")
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of results to show from each search",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=DEFAULT_CANDIDATE_K,
        help="Number of dense candidates for hybrid reranking",
    )
    parser.add_argument(
        "--filter",
        default=None,
        help='Optional Milvus filter expression, example: domain == "telecom"',
    )
    parser.add_argument(
        "--semantic-collection-name",
        default=SEMANTIC_COLLECTION_NAME,
        help="Milvus collection name for semantic search",
    )
    parser.add_argument(
        "--hybrid-collection-name",
        default=HYBRID_COLLECTION_NAME,
        help="Milvus collection name for hybrid search",
    )

    args = parser.parse_args()

    compare_semantic_vs_hybrid(
        query=args.query,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        semantic_collection_name=args.semantic_collection_name,
        hybrid_collection_name=args.hybrid_collection_name,
        filter_expr=args.filter,
    )


if __name__ == "__main__":
    main()
