import json
import os
from urllib import error, request

from dotenv import load_dotenv


OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"


def load_openai_settings() -> tuple[str, str]:
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("OPENAI_BASE_MODEL", "").strip()

    return api_key, model


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


def build_relevance_context(chunks: list[dict]) -> str:
    lines = []

    for index, chunk in enumerate(chunks, start=1):
        entity = chunk["entity"]
        lines.append(f"Chunk {index}")
        lines.append(f"source_file: {entity.get('source_file', '')}")
        lines.append(f"chunk_index: {entity.get('chunk_index', '')}")
        lines.append(f"title: {entity.get('title', '')}")
        lines.append(f"domain: {entity.get('domain', '')}")
        lines.append(f"system_name: {entity.get('system_name', '')}")
        lines.append(f"doc_type: {entity.get('doc_type', '')}")
        lines.append(f"text: {entity.get('chunk_text', '')}")
        lines.append("")

    return "\n".join(lines).strip()


def parse_judge_response(text: str) -> dict:
    data = json.loads(text)

    decision = str(data.get("decision", "")).strip().upper()
    reason = str(data.get("reason", "")).strip()
    confidence = str(data.get("confidence", "")).strip().lower()

    if decision not in {"PASS", "REJECT"}:
        raise ValueError("Invalid decision from relevance judge.")
    if confidence not in {"high", "medium", "low"}:
        raise ValueError("Invalid confidence from relevance judge.")

    return {
        "decision": decision,
        "reason": reason,
        "confidence": confidence,
    }


def judge_relevance(user_query: str, chunks: list[dict]) -> dict:
    if not chunks:
        return {
            "decision": "REJECT",
            "reason": "No retrieved context was available.",
            "confidence": "high",
        }

    api_key, model = load_openai_settings()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing.")
    if not model:
        raise RuntimeError("OPENAI_BASE_MODEL is missing.")

    context = build_relevance_context(chunks)

    instructions = (
        "You are a strict relevance judge for retrieval-augmented generation. "
        "Answer the question: Does the retrieved context directly help answer the user query? "
        "Return only strict JSON with keys decision, reason, confidence. "
        "decision must be PASS or REJECT. confidence must be high, medium, or low. "
        "If the query topic and retrieved context topic clearly mismatch, decision must be REJECT. "
        "If retrieved context is insufficient, decision must be REJECT. "
        "Be conservative."
    )

    user_prompt = (
        f"User query:\n{user_query}\n\n"
        f"Retrieved context:\n{context}\n\n"
        "Return strict JSON only."
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
        raise RuntimeError(f"OpenAI relevance judge request failed: {exc.code} {exc.reason}\n{error_body}") from exc

    judge_text = extract_response_text(payload)
    if not judge_text:
        raise RuntimeError("Relevance judge response did not contain text.")

    return parse_judge_response(judge_text)
