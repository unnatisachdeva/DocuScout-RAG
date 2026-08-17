

from web_loader.crawler import crawl_website
from ingestion import create_vectorstore

website_docs= crawl_website("https://www.thexgroup.ca",
    max_pages=20
)
print("\nWebsite documents:", len(website_docs))

vectorstore, num_docs, num_chunks = create_vectorstore(
    website_docs,
    persist_directory="test_web_vector_store"
)

print("\n==========================")
print("Ingestion completed!")
print("Documents:", num_docs)
print("Chunks:", num_chunks)
print("==========================")