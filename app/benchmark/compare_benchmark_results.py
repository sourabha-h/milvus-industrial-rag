import csv
from collections import defaultdict
from pathlib import Path


RESULTS_FILE = Path("data/benchmark/results/search_benchmark_results.csv")


def get_first_present(row: dict[str, str], keys: list[str], default: str = "") -> str:
    for key in keys:
        value = row.get(key, "")
        if value != "":
            return value
    return default


def parse_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value else 0.0


def load_latest_rows(results_file: Path):
    latest_rows = {}

    with results_file.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            collection_name = get_first_present(row, ["collection_name"])
            query_name = get_first_present(row, ["query_name", "test_name"])
            timestamp = get_first_present(row, ["timestamp"])

            if not collection_name or not query_name:
                continue

            key = (collection_name, query_name)
            current = latest_rows.get(key)

            if current is None or timestamp >= current["timestamp"]:
                latest_rows[key] = {
                    "timestamp": timestamp,
                    "collection_name": collection_name,
                    "query_name": query_name,
                    "filter_expr": get_first_present(row, ["filter_expr", "filter"]),
                    "row_count": get_first_present(row, ["row_count"]),
                    "iterations": get_first_present(row, ["iterations"]),
                    "avg_ms": parse_float(row, "avg_ms"),
                    "p50_ms": parse_float(row, "p50_ms"),
                    "p95_ms": parse_float(row, "p95_ms"),
                    "max_ms": parse_float(row, "max_ms"),
                }

    return latest_rows


def compare_benchmark_results():
    if not RESULTS_FILE.exists():
        print(f"Error: CSV file not found: {RESULTS_FILE}")
        return

    latest_rows = load_latest_rows(RESULTS_FILE)

    if not latest_rows:
        print(f"No benchmark rows found in: {RESULTS_FILE}")
        return

    grouped = defaultdict(list)
    for row in latest_rows.values():
        grouped[row["query_name"]].append(row)

    avg_wins = defaultdict(int)
    p95_wins = defaultdict(int)

    for query_name in sorted(grouped):
        rows = grouped[query_name]
        best_avg_row = min(rows, key=lambda row: row["avg_ms"])
        best_p95_row = min(rows, key=lambda row: row["p95_ms"])

        avg_wins[best_avg_row["collection_name"]] += 1
        p95_wins[best_p95_row["collection_name"]] += 1

        print(f"\nQuery: {query_name}")
        for row in sorted(rows, key=lambda item: item["collection_name"]):
            print(
                f"  {row['collection_name']}: "
                f"avg_ms={row['avg_ms']:.2f}, "
                f"p50_ms={row['p50_ms']:.2f}, "
                f"p95_ms={row['p95_ms']:.2f}, "
                f"max_ms={row['max_ms']:.2f}"
            )

        print(f"  best_avg: {best_avg_row['collection_name']}")
        print(f"  best_p95: {best_p95_row['collection_name']}")

    print("\nOverall Summary")
    print("avg_ms wins:")
    for collection_name in sorted(avg_wins):
        print(f"  {collection_name}: {avg_wins[collection_name]}")

    print("p95_ms wins:")
    for collection_name in sorted(p95_wins):
        print(f"  {collection_name}: {p95_wins[collection_name]}")


if __name__ == "__main__":
    compare_benchmark_results()
