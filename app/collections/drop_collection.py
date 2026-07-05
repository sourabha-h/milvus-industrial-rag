from app.db.milvus_client import get_milvus_client

COLLECTION_NAME = "industrial_documents"


def drop_collection():
    client = get_milvus_client()

    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)
        print(f"Dropped collection: {COLLECTION_NAME}")
    else:
        print(f"Collection does not exist: {COLLECTION_NAME}")


if __name__ == "__main__":
    drop_collection()
