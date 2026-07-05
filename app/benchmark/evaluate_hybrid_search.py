import argparse

from app.search.compare_semantic_vs_hybrid import (
    DEFAULT_CANDIDATE_K,
    HYBRID_COLLECTION_NAME,
    SEMANTIC_COLLECTION_NAME,
    run_hybrid_search,
    run_semantic_search,
)


QUERIES = [
    "ORA-12541 listener error",
    "voucher recharge failed",
    "oracle listener restart service",
    "kubernetes pod restart",
    "database connection timeout",
]

DEFAULT_TOP_K = 5


def format_reference(row: dict) -> str:
    return f"{row.get('title', '')} | {row.get('source_file', '')}"


def evaluate_hybrid_search(
    collection_name: str = HYBRID_COLLECTION_NAME,
    top_k: int = DEFAULT_TOP_K,
    candidate_k: int = 50,
    filter_expr: str | None = None,
):
    for query in QUERIES:
        semantic_rows = run_semantic_search(query, SEMANTIC_COLLECTION_NAME, top_k)
        hybrid_rows = run_hybrid_search(
            query=query,
            collection_name=collection_name,
            top_k=top_k,
            candidate_k=candidate_k,
            filter_expr=filter_expr,
        )

        semantic_refs = [format_reference(row) for row in semantic_rows]
        hybrid_refs = [format_reference(row) for row in hybrid_rows]
        ranking_changed = semantic_refs != hybrid_refs

        print(f"\nQuery: {query}")
        if filter_expr:
            print(f"Filter: {filter_expr}")
        print(f"Ranking changed: {'Yes' if ranking_changed else 'No'}")

        print("Semantic top results:")
        if semantic_rows:
            for rank, row in enumerate(semantic_rows[:top_k], start=1):
                print(f"  {rank}. {row.get('title', '')} | {row.get('source_file', '')}")
        else:
            print("  None")

        print("Hybrid top results:")
        if hybrid_rows:
            for rank, row in enumerate(hybrid_rows[:top_k], start=1):
                matched_keywords = row.get("matched_keywords") or []
                matched_text = ", ".join(matched_keywords) if matched_keywords else "None"
                print(f"  {rank}. {row.get('title', '')} | {row.get('source_file', '')}")
                print(
                    f"     final_score={row.get('final_score', 0.0):.4f} "
                    f"vector_score={row.get('vector_score', 0.0):.4f} "
                    f"keyword_boost={row.get('keyword_boost', 0.0):.4f}"
                )
                print(f"     matched_keywords: {matched_text}")
        else:
            print("  None")


def main():
    parser = argparse.ArgumentParser(description="Evaluate semantic vs hybrid search")
    parser.add_argument(
        "--collection-name",
        default=HYBRID_COLLECTION_NAME,
        help="Milvus collection name for hybrid search",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of results to show per query",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=50,
        help="Number of dense candidates for hybrid reranking",
    )
    parser.add_argument(
        "--filter",
        default=None,
        help='Optional Milvus filter expression, example: domain == "telecom"',
    )

    args = parser.parse_args()

    evaluate_hybrid_search(
        collection_name=args.collection_name,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        filter_expr=args.filter,
    )


if __name__ == "__main__":
    main()
