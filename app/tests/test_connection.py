from app.db.milvus_client import get_milvus_client


def main():
    client = get_milvus_client()

    collections = client.list_collections()

    print("Milvus connection successful.")
    print("Existing collections:", collections)


if __name__ == "__main__":
    main()
