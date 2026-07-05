import argparse
import io
import json
import os
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from urllib import error, request

from dotenv import load_dotenv

from app.search.hybrid_search import hybrid_search
from app.rag.relevance_guardrail import judge_relevance


DEFAULT_COLLECTION_NAME = "industrial_documents_hnsw"
DEFAULT_TOP_K = 3
DEFAULT_CANDIDATE_K = 50
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
RAG_AUDIT_LOG_PATH = Path("data/rag/rag_audit_log.jsonl")


def load_openai_settings() -> tuple[str, str]:
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("OPENAI_BASE_MODEL", "").strip()

    return api_key, model


def collect_hybrid_chunks(query: str, collection_name: str, top_k: int, candidate_k: int, filter_expr: str | None):
    buffer = io.StringIO()

    with redirect_stdout(buffer):
        results = hybrid_search(
            query=query,
            collection_name=collection_name,
            filter_expr=filter_expr,
            top_k=top_k,
            candidate_k=candidate_k,
        )

    return results


def build_context(chunks: list[dict]) -> str:
    if not chunks:
        return ""

    lines = []

    for index, chunk in enumerate(chunks, start=1):
        entity = chunk["entity"]
        lines.append(f"Chunk {index}")
        lines.append(f"source_file: {entity.get('source_file', '')}")
        lines.append(f"chunk_index: {entity.get('chunk_index', '')}")
        lines.append(f"title: {entity.get('title', '')}")
        lines.append(f"text: {entity.get('chunk_text', '')}")
        lines.append("")

    return "\n".join(lines).strip()


def extract_response_text(payload: dict) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    texts = []

    for item in payload.get("output", []):
        item_type = item.get("type")

        if item_type == "message":
            for content in item.get("content", []):
                text = content.get("text")
                if text:
                    texts.append(text)
        elif item_type == "output_text":
            text = item.get("text")
            if text:
                texts.append(text)

    return "\n".join(texts).strip()


def generate_answer(api_key: str, model: str, query: str, context: str) -> str:
    instructions = (
        "You are a grounded industrial support assistant. Answer the user using ONLY the provided context. "
        "If the retrieved context is weak, missing, or unrelated, respond exactly with: "
        "\"Retrieved context is insufficient to answer confidently.\" "
        "Do not use outside knowledge. "
        "Format the answer with these sections and keep them concise: "
        "Summary, Evidence Used, Recommended Actions, Confidence / Limitations. "
        "In Evidence Used, cite only the provided chunks by source_file and chunk_index."
    )

    user_prompt = (
        f"User query:\n{query}\n\n"
        f"Retrieved context:\n{context}\n\n"
        "Write the answer using only the retrieved context and the required section structure."
    )

    body = json.dumps(
        {
            "model": model,
            "instructions": instructions,
            "input": user_prompt,
        }
    ).encode("utf-8")

    req = request.Request(
        OPENAI_RESPONSES_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"OpenAI request failed: {exc.code} {exc.reason}\n{error_body}") from exc

    answer = extract_response_text(payload)
    if not answer:
        raise RuntimeError("OpenAI response did not contain answer text.")

    return answer


def append_rag_audit_log(
    query: str,
    collection_name: str,
    top_k: int,
    candidate_k: int,
    filter_expr: str | None,
    chunks: list[dict],
    guardrail: dict,
    final_status: str,
):
    RAG_AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "query": query,
        "collection_name": collection_name,
        "top_k": top_k,
        "candidate_k": candidate_k,
        "filter": filter_expr or "",
        "retrieved_sources": [
            {
                "source_file": chunk["entity"].get("source_file", ""),
                "chunk_index": chunk["entity"].get("chunk_index", ""),
            }
            for chunk in chunks
        ],
        "guardrail_decision": guardrail["decision"],
        "guardrail_reason": guardrail["reason"],
        "guardrail_confidence": guardrail["confidence"],
        "final_status": final_status,
    }

    with RAG_AUDIT_LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def basic_rag(
    query: str,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    top_k: int = DEFAULT_TOP_K,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    filter_expr: str | None = None,
):
    api_key, model = load_openai_settings()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing.")
    if not model:
        raise RuntimeError("OPENAI_BASE_MODEL is missing.")

    chunks = collect_hybrid_chunks(query, collection_name, top_k, candidate_k, filter_expr)
    context = build_context(chunks)

    print(f"\nUser Query: {query}")
    print(f"Collection: {collection_name}")

    if filter_expr:
        print(f"Filter: {filter_expr}")

    print("Sources Used:")
    if chunks:
        for chunk in chunks:
            entity = chunk["entity"]
            print(f"- {entity.get('source_file', '')} | chunk_index={entity.get('chunk_index', '')}")
    else:
        print("- None")

    guardrail = judge_relevance(query, chunks)
    print(f"Guardrail Decision: {guardrail['decision']}")
    print(f"Guardrail Reason: {guardrail['reason']}")
    print(f"Guardrail Confidence: {guardrail['confidence']}")

    if guardrail["decision"] == "REJECT":
        append_rag_audit_log(
            query=query,
            collection_name=collection_name,
            top_k=top_k,
            candidate_k=candidate_k,
            filter_expr=filter_expr,
            chunks=chunks,
            guardrail=guardrail,
            final_status="REJECTED_BY_GUARDRAIL",
        )
        print("\nGenerated Answer:")
        print("Retrieved context is insufficient to answer confidently.")
        return

    if not context:
        append_rag_audit_log(
            query=query,
            collection_name=collection_name,
            top_k=top_k,
            candidate_k=candidate_k,
            filter_expr=filter_expr,
            chunks=chunks,
            guardrail=guardrail,
            final_status="REJECTED_BY_GUARDRAIL",
        )
        print("\nGenerated Answer:")
        print("Retrieved context is insufficient to answer confidently.")
        return

    answer = generate_answer(api_key=api_key, model=model, query=query, context=context)
    append_rag_audit_log(
        query=query,
        collection_name=collection_name,
        top_k=top_k,
        candidate_k=candidate_k,
        filter_expr=filter_expr,
        chunks=chunks,
        guardrail=guardrail,
        final_status="ANSWER_GENERATED",
    )

    print("\nGenerated Answer:")
    print(answer)


def main():
    parser = argparse.ArgumentParser(description="Basic RAG answer generation using Milvus hybrid search")
    parser.add_argument("--query", required=True, help="User question to answer")
    parser.add_argument(
        "--collection-name",
        default=DEFAULT_COLLECTION_NAME,
        help="Milvus collection name to search",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of retrieved chunks to use",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=DEFAULT_CANDIDATE_K,
        help="Number of dense candidates before reranking",
    )
    parser.add_argument(
        "--filter",
        default=None,
        help='Optional Milvus filter expression, example: domain == "telecom"',
    )

    args = parser.parse_args()

    basic_rag(
        query=args.query,
        collection_name=args.collection_name,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        filter_expr=args.filter,
    )


if __name__ == "__main__":
    main()
