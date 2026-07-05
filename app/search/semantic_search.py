from app.db.milvus_client import get_milvus_client
from app.embeddings.embedding_service import EmbeddingService


COLLECTION_NAME = "industrial_documents"


def semantic_search(query: str, top_k: int = 3):
    client = get_milvus_client()
    embedder = EmbeddingService()

    query_vector = embedder.embed_text(query)

    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        anns_field="embedding",
        limit=top_k,
        output_fields=[
            "doc_id",
            "title",
            "chunk_text",
            "domain",
            "system_name",
            "doc_type",
            "created_date",
            "source_file",
            "source_type",
            "chunk_index",
            "page_number",
            "ingested_at",
            "content_hash",
        ],
    )

    print(f"\nQuery: {query}")
    print("=" * 80)

    for hits in results:
        for rank, hit in enumerate(hits, start=1):
            entity = hit["entity"]

            print(f"\nRank: {rank}")
            print(f"Score: {hit['distance']}")
            print(f"Title: {entity['title']}")
            print(f"Doc ID: {entity['doc_id']}")
            print(f"Domain: {entity['domain']}")
            print(f"System: {entity['system_name']}")
            print(f"Doc Type: {entity['doc_type']}")
            print(f"Text: {entity['chunk_text']}")
            print(f"Source File: {entity['source_file']}")
            print(f"Source Type: {entity['source_type']}")
            print(f"Chunk Index: {entity['chunk_index']}")
            print(f"Page Number: {entity['page_number']}")
            print(f"Ingested At: {entity['ingested_at']}")
            print(f"Content Hash: {entity['content_hash']}")


if __name__ == "__main__":
    semantic_search("database listener service not found", top_k=1)
