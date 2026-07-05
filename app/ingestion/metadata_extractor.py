from pathlib import Path


HEADER_ALIASES = {
    "title": ["title"],
    "domain": ["domain"],
    "system_name": ["system", "system_name", "system name"],
    "doc_type": ["document type", "doc_type", "doc type"],
}


def _normalize_key(key: str) -> str:
    return key.strip().lower().replace("_", " ")


def _extract_header_metadata(text: str) -> dict:
    """
    Extract simple metadata headers from the first few lines.

    Supported examples:
    Title: Oracle Listener Connection Failure
    Domain: database
    System: Oracle
    Document Type: SOP
    """
    metadata = {}

    lines = text.splitlines()[:20]

    for line in lines:
        if ":" not in line:
            continue

        raw_key, raw_value = line.split(":", 1)
        key = _normalize_key(raw_key)
        value = raw_value.strip()

        if not value:
            continue

        for target_key, aliases in HEADER_ALIASES.items():
            if key in aliases:
                metadata[target_key] = value
                break

    return metadata


def _infer_metadata_from_keywords(file_path: Path, text: str) -> dict:
    lower_path = str(file_path).lower()
    lower_text = text.lower()

    domain = "general"
    system_name = "Unknown"
    doc_type = "Document"

    if "oracle" in lower_path or "oracle" in lower_text or "ora-" in lower_text:
        domain = "database"
        system_name = "Oracle"

    elif "postgres" in lower_path or "postgresql" in lower_text:
        domain = "database"
        system_name = "PostgreSQL"

    elif "mysql" in lower_path or "mysql" in lower_text:
        domain = "database"
        system_name = "MySQL"

    elif "voucher" in lower_path or "recharge" in lower_text:
        domain = "telecom"
        system_name = "Voucher"

    elif "billing" in lower_path or "billing" in lower_text:
        domain = "telecom"
        system_name = "Billing"

    elif "smpp" in lower_path or "smsc" in lower_text:
        domain = "telecom"
        system_name = "SMSC"

    elif "linux" in lower_path or "filesystem" in lower_text:
        domain = "infrastructure"
        system_name = "Linux"

    elif "kubernetes" in lower_path or "crashloopbackoff" in lower_text:
        domain = "cloud"
        system_name = "Kubernetes"

    elif "docker" in lower_path or "docker" in lower_text:
        domain = "cloud"
        system_name = "Docker"

    elif "firewall" in lower_path or "firewall" in lower_text:
        domain = "network"
        system_name = "Firewall"

    elif "router" in lower_path or "router" in lower_text:
        domain = "network"
        system_name = "Router"

    elif "iam" in lower_path or "waf" in lower_text or "siem" in lower_text:
        domain = "security"
        system_name = "Security"

    if "rca" in lower_path or "root cause" in lower_text:
        doc_type = "RCA"
    elif "sop" in lower_path:
        doc_type = "SOP"
    elif "troubleshooting" in lower_path:
        doc_type = "Troubleshooting"
    elif "incident" in lower_path:
        doc_type = "Incident"

    title = file_path.stem.replace("_", " ").replace("-", " ").title()

    return {
        "title": title,
        "domain": domain,
        "system_name": system_name,
        "doc_type": doc_type,
    }


def infer_metadata(file_path: Path, text: str) -> dict:
    """
    Prefer explicit document headers.
    Fallback to keyword-based metadata inference.
    """
    fallback_metadata = _infer_metadata_from_keywords(file_path, text)
    header_metadata = _extract_header_metadata(text)

    return {
        "title": _truncate(header_metadata.get("title", fallback_metadata["title"]), 500),
        "domain": _truncate(header_metadata.get("domain", fallback_metadata["domain"]), 100),
        "system_name": _truncate(
            header_metadata.get("system_name", fallback_metadata["system_name"]), 100
        ),
        "doc_type": _truncate(header_metadata.get("doc_type", fallback_metadata["doc_type"]), 100),
    }

def _truncate(value: str, max_length: int) -> str:
    return value[:max_length]
