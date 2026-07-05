import csv
import argparse
import statistics
import time
from datetime import datetime
from pathlib import Path

from app.db.milvus_client import get_milvus_client
from app.embeddings.embedding_service import EmbeddingService


COLLECTION_NAME = "industrial_documents"

RESULTS_DIR = Path("data/benchmark/results")
RESULTS_FILE = RESULTS_DIR / "search_benchmark_results.csv"

OUTPUT_FIELDS = [
    "doc_id",
    "source_file",
    "title",
    "domain",
    "system_name",
    "doc_type",
    "chunk_index",
]


TEST_QUERIES = [
    {
        "name": "voucher failure",
        "query": "voucher recharge failure request did not reach platform",
        "filter": None,
    },
    {
        "name": "oracle listener",
        "query": "database listener error connection timeout",
        "filter": None,
    },
    {
        "name": "kubernetes restart",
        "query": "pod restart loop service unavailable",
        "filter": None,
    },
    {
        "name": "telecom filtered",
        "query": "transaction failure in charging or billing platform",
        "filter": 'domain == "telecom"',
    },
    {
        "name": "database filtered",
        "query": "database connection timeout listener error",
        "filter": 'domain == "database"',
    },
]


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0

    sorted_values = sorted(values)
    index = int((len(sorted_values) - 1) * p)
    return sorted_values[index]


def run_single_search(client, collection_name: str, query_vector, filter_expr: str | None, top_k: int = 5):
    search_kwargs = {
        "collection_name": collection_name,
        "data": [query_vector],
        "anns_field": "embedding",
        "limit": top_k,
        "output_fields": OUTPUT_FIELDS,
    }

    if filter_expr:
        search_kwargs["filter"] = filter_expr

    return client.search(**search_kwargs)


def open_csv_writer():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    file_exists = RESULTS_FILE.exists()

    csv_file = RESULTS_FILE.open("a", newline="", encoding="utf-8")

    writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "timestamp",
            "collection_name",
            "row_count",
            "iterations",
            "test_name",
            "query",
            "filter",
            "result_count",
            "min_ms",
            "avg_ms",
            "p50_ms",
            "p95_ms",
            "max_ms",
        ],
    )

    if not file_exists:
        writer.writeheader()

    return csv_file, writer


def benchmark_search(collection_name: str = COLLECTION_NAME, iterations: int = 10):
    client = get_milvus_client()
    embedder = EmbeddingService()

    collection_stats = client.get_collection_stats(collection_name=collection_name)
    row_count = collection_stats.get("row_count", "unknown")

    print(f"Running benchmark on collection: {collection_name}")
    print(f"Collection row count: {row_count}")
    print(f"Iterations per query: {iterations}")
    print("=" * 100)

    csv_file, writer = open_csv_writer()

    try:
        for test in TEST_QUERIES:
            query = test["query"]
            filter_expr = test["filter"]

            query_vector = embedder.embed_text(query)

            latencies_ms = []

            # Warmup
            run_single_search(client, collection_name, query_vector, filter_expr)

            for _ in range(iterations):
                start = time.perf_counter()
                results = run_single_search(client, collection_name, query_vector, filter_expr)
                end = time.perf_counter()

                latencies_ms.append((end - start) * 1000)

            result_count = len(results[0]) if results else 0

            min_ms = min(latencies_ms)
            avg_ms = statistics.mean(latencies_ms)
            p50_ms = statistics.median(latencies_ms)
            p95_ms = percentile(latencies_ms, 0.95)
            max_ms = max(latencies_ms)

            print(f"\nTest: {test['name']}")
            print(f"Query: {query}")
            print(f"Filter: {filter_expr}")
            print(f"Result Count: {result_count}")
            print(f"Min latency: {min_ms:.2f} ms")
            print(f"Avg latency: {avg_ms:.2f} ms")
            print(f"P50 latency: {p50_ms:.2f} ms")
            print(f"P95 latency: {p95_ms:.2f} ms")
            print(f"Max latency: {max_ms:.2f} ms")

            writer.writerow(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "collection_name": collection_name,
                    "row_count": row_count,
                    "iterations": iterations,
                    "test_name": test["name"],
                    "query": query,
                    "filter": filter_expr or "",
                    "result_count": result_count,
                    "min_ms": round(min_ms, 2),
                    "avg_ms": round(avg_ms, 2),
                    "p50_ms": round(p50_ms, 2),
                    "p95_ms": round(p95_ms, 2),
                    "max_ms": round(max_ms, 2),
                }
            )

    finally:
        csv_file.close()

    print(f"\nBenchmark results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Milvus search benchmark")
    parser.add_argument(
        "--collection-name",
        default=COLLECTION_NAME,
        help="Milvus collection name to benchmark",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of benchmark iterations per query",
    )

    args = parser.parse_args()
    benchmark_search(collection_name=args.collection_name, iterations=args.iterations)
