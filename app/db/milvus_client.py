from pymilvus import MilvusClient
from app.core.config import MILVUS_HOST, MILVUS_PORT, MILVUS_DB_NAME


def get_milvus_client() -> MilvusClient:
    uri = f"http://{MILVUS_HOST}:{MILVUS_PORT}"

    client = MilvusClient(
        uri=uri,
        db_name=MILVUS_DB_NAME,
    )

    return client
