from web_loader.crawler import crawl_website

documents= crawl_website("https://www.thexgroup.ca",
    max_pages=20)
print("\n==========================")
print("Documents returned:", len(documents))
print("==========================")

for document in documents:

    print("\n------------------------")

    print(
        "URL:",
        document.metadata["source"]
    )

    print(
        "Title:",
        document.metadata["title"]
    )

    print(
        "Text:",
        document.page_content[:300]
    )