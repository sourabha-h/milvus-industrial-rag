from app.db.milvus_client import get_milvus_client
from app.embeddings.embedding_service import EmbeddingService


COLLECTION_NAME = "industrial_documents"


SAMPLE_DOCUMENTS = [
    {
        "id": 1,
        "doc_id": "DOC-ORACLE-001",
        "title": "Oracle Listener Connection Failure",
        "chunk_text": (
            "Oracle database connection failed with ORA-12514. "
            "The listener does not currently know of service requested in connect descriptor. "
            "Check listener.ora, tnsnames.ora, service registration, database instance status, "
            "and network connectivity between application server and database server."
        ),
        "domain": "database",
        "system_name": "Oracle",
        "doc_type": "SOP",
        "created_date": "2026-06-01",
    },
    {
        "id": 2,
        "doc_id": "DOC-VOUCHER-001",
        "title": "Voucher Recharge Failure Investigation",
        "chunk_text": (
            "Voucher recharge failed although the voucher serial number was valid. "
            "Investigation should check SMF voucher reference, VWS voucher reference, "
            "subscriber MSISDN, redemption date, EDR files, and CDR recharge failure cause. "
            "If voucher is not redeemed and no request is found in CDR, the request may not have reached the platform."
        ),
        "domain": "telecom",
        "system_name": "Voucher",
        "doc_type": "RCA",
        "created_date": "2026-06-02",
    },
    {
        "id": 3,
        "doc_id": "DOC-LINUX-001",
        "title": "Linux Disk Full Incident",
        "chunk_text": (
            "Application outage occurred because the Linux server root filesystem reached 100 percent usage. "
            "Check large log files, rotate application logs, clean temporary files, validate disk growth trend, "
            "and configure alert threshold for filesystem utilization."
        ),
        "domain": "infrastructure",
        "system_name": "Linux",
        "doc_type": "Incident",
        "created_date": "2026-06-03",
    },
    {
        "id": 4,
        "doc_id": "DOC-SMPP-001",
        "title": "SMPP Bind Failure",
        "chunk_text": (
            "SMS delivery failed due to SMPP bind failure between application and SMSC. "
            "Check bind credentials, source IP whitelist, bind mode, enquire_link response, "
            "SMSC connectivity, and throttling configuration."
        ),
        "domain": "telecom",
        "system_name": "SMSC",
        "doc_type": "Troubleshooting",
        "created_date": "2026-06-04",
    },
    {
        "id": 5,
        "doc_id": "DOC-K8S-001",
        "title": "Kubernetes Pod CrashLoopBackOff",
        "chunk_text": (
            "Kubernetes pod entered CrashLoopBackOff state after deployment. "
            "Check container logs, environment variables, image version, readiness probe, liveness probe, "
            "resource limits, config map, secret values, and application startup errors."
        ),
        "domain": "cloud",
        "system_name": "Kubernetes",
        "doc_type": "SOP",
        "created_date": "2026-06-05",
    },
    {
        "id": 6,
        "doc_id": "DOC-BILLING-001",
        "title": "Billing System Timeout",
        "chunk_text": (
            "Billing transaction timeout occurred during customer balance update. "
            "Check application logs, database response time, downstream charging engine latency, "
            "thread pool saturation, connection pool utilization, and retry configuration."
        ),
        "domain": "telecom",
        "system_name": "Billing",
        "doc_type": "RCA",
        "created_date": "2026-06-06",
    },
]


def insert_sample_data():
    client = get_milvus_client()
    embedder = EmbeddingService()

    texts = [doc["chunk_text"] for doc in SAMPLE_DOCUMENTS]
    embeddings = embedder.embed_texts(texts)

    rows = []

    for doc, embedding in zip(SAMPLE_DOCUMENTS, embeddings):
        row = {
            **doc,
            "embedding": embedding,
        }
        rows.append(row)

    result = client.insert(
        collection_name=COLLECTION_NAME,
        data=rows,
    )

    print("Inserted sample documents.")
    print(result)


if __name__ == "__main__":
    insert_sample_data()
