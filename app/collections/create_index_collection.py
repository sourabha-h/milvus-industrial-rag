import argparse

from pymilvus import DataType

from app.db.milvus_client import get_milvus_client


VECTOR_DIM = 384


def build_index_params(client, index_type: str):
    index_type = index_type.upper()

    index_params = client.prepare_index_params()

    if index_type == "AUTOINDEX":
        index_params.add_index(
            field_name="embedding",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )

    elif index_type == "HNSW":
        index_params.add_index(
            field_name="embedding",
            index_type="HNSW",
            metric_type="COSINE",
            params={
                "M": 16,
                "efConstruction": 200,
            },
        )

    elif index_type == "IVF_FLAT":
        index_params.add_index(
            field_name="embedding",
            index_type="IVF_FLAT",
            metric_type="COSINE",
            params={
                "nlist": 128,
            },
        )

    else:
        raise ValueError(
            f"Unsupported index type: {index_type}. "
            "Supported: AUTOINDEX, HNSW, IVF_FLAT"
        )

    return index_params


def create_schema(client):
    schema = client.create_schema(
        auto_id=False,
        enable_dynamic_field=False,
    )

    schema.add_field(
        field_name="id",
        datatype=DataType.INT64,
        is_primary=True,
    )

    schema.add_field(
        field_name="doc_id",
        datatype=DataType.VARCHAR,
        max_length=100,
    )

    schema.add_field(
        field_name="title",
        datatype=DataType.VARCHAR,
        max_length=500,
    )

    schema.add_field(
        field_name="chunk_text",
        datatype=DataType.VARCHAR,
        max_length=5000,
    )

    schema.add_field(
        field_name="domain",
        datatype=DataType.VARCHAR,
        max_length=100,
    )

    schema.add_field(
        field_name="system_name",
        datatype=DataType.VARCHAR,
        max_length=100,
    )

    schema.add_field(
        field_name="doc_type",
        datatype=DataType.VARCHAR,
        max_length=100,
    )

    schema.add_field(
        field_name="created_date",
        datatype=DataType.VARCHAR,
        max_length=20,
    )

    schema.add_field(
        field_name="source_file",
        datatype=DataType.VARCHAR,
        max_length=1000,
    )

    schema.add_field(
        field_name="source_type",
        datatype=DataType.VARCHAR,
        max_length=50,
    )

    schema.add_field(
        field_name="chunk_index",
        datatype=DataType.INT64,
    )

    schema.add_field(
        field_name="page_number",
        datatype=DataType.INT64,
    )

    schema.add_field(
        field_name="ingested_at",
        datatype=DataType.VARCHAR,
        max_length=40,
    )

    schema.add_field(
        field_name="content_hash",
        datatype=DataType.VARCHAR,
        max_length=128,
    )

    schema.add_field(
        field_name="embedding",
        datatype=DataType.FLOAT_VECTOR,
        dim=VECTOR_DIM,
    )

    return schema


def create_collection(collection_name: str, index_type: str):
    client = get_milvus_client()

    if client.has_collection(collection_name):
        print(f"Collection already exists: {collection_name}")
        return

    schema = create_schema(client)
    index_params = build_index_params(client, index_type)

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
    )

    print(f"Collection created: {collection_name}")
    print(f"Index type: {index_type.upper()}")


def main():
    parser = argparse.ArgumentParser(
        description="Create Milvus industrial_documents collection with selected index type"
    )

    parser.add_argument(
        "--collection-name",
        required=True,
        help="Collection name to create",
    )

    parser.add_argument(
        "--index-type",
        required=True,
        choices=["AUTOINDEX", "HNSW", "IVF_FLAT"],
        help="Vector index type",
    )

    args = parser.parse_args()

    create_collection(
        collection_name=args.collection_name,
        index_type=args.index_type,
    )


if __name__ == "__main__":
    main()
