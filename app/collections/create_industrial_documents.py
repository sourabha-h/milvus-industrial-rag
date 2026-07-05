from pymilvus import DataType
from app.db.milvus_client import get_milvus_client


COLLECTION_NAME = "industrial_documents"
VECTOR_DIM = 384


def create_collection():
    client = get_milvus_client()

    if client.has_collection(COLLECTION_NAME):
        print(f"Collection already exists: {COLLECTION_NAME}")
        return

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

    index_params = client.prepare_index_params()

    index_params.add_index(
        field_name="embedding",
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )

    print(f"Collection created successfully: {COLLECTION_NAME}")


if __name__ == "__main__":
    create_collection()
