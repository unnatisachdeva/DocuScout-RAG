
# Role: Finds web pages and turns them into Documents.

from web_loader.playwright_scraper import get_page_with_playwright
from web_loader.web_cleaner import clean_webpage
import requests
from bs4 import BeautifulSoup
from urllib.parse import urldefrag, urljoin, urlparse
from collections import deque

from langchain_core.documents import Document
from web_loader.web_cleaner import clean_webpage


headers = {
    "User-Agent": "Mozilla/5.0"
}


# --------------------------------
# URL normalization
# --------------------------------

def normalize_url(url):

    # Remove fragment
    url, fragment = urldefrag(url)

    parsed = urlparse(url)

    # Remove trailing slash
    path = parsed.path.rstrip("/")

    # Remove query parameters
    normalized_url = parsed._replace(
        path=path,
        query=""
    ).geturl()

    return normalized_url


# --------------------------------
# Content usability check
# --------------------------------

def is_content_usable(soup):

    # Check visible text
    text = soup.get_text(
        " ",
        strip=True
    )

    if len(text) < 200:
        return False

    # Check number of links
    links = soup.find_all("a")

    if len(links) == 0:
        return False

    return True


# --------------------------------
# Main crawler function
# --------------------------------

def crawl_website(start_url, max_pages=20):

    # Normalize starting URL
    start_url = normalize_url(start_url)

    # Domain of website we want to crawl
    base_domain = urlparse(start_url).netloc

    # URLs waiting to be visited
    to_visit = deque([start_url])

    # URLs already visited
    visited = set()

    # LangChain Documents
    documents = []


    # --------------------------------
    # Crawler loop
    # --------------------------------

    while to_visit and len(visited) < max_pages:

        current_url = to_visit.popleft()

        # Don't visit same URL twice
        if current_url in visited:
            continue

        print("Visiting:", current_url)


        # --------------------------------
        # Try Requests first
        # --------------------------------

        try:

            response = requests.get(
                current_url,
                headers=headers,
                timeout=10
            )

            print(
                "Status:",
                response.status_code
            )

        except requests.RequestException as e:

            print("Error:", e)

            continue


        # Mark as visited
        visited.add(current_url)


        # --------------------------------
        # Status handling
        # --------------------------------

        if response.status_code == 404:

            print("Page not found")

            continue


        if response.status_code == 403:

            print("Access forbidden")

            continue


        if response.status_code == 429:

            print("Too many requests")

            continue


        if response.status_code >= 400:

            print("Request failed")

            continue


        # --------------------------------
        # Content type
        # --------------------------------

        content_type = response.headers.get(
            "Content-Type",
            ""
        ).lower()

        print(
            "Content-Type:",
            content_type
        )


        # Only process HTML
        if "text/html" not in content_type:

            print(
                "Skipping non-HTML resource"
            )

            continue


        # --------------------------------
        # Parse Requests HTML
        # --------------------------------

        html = response.text

        soup = BeautifulSoup(
            html,
            "html.parser"
        )

        # --------------------------------
        # Check content
        # --------------------------------

        if not is_content_usable(soup):

            print(
                "Content seems incomplete"
            )

            print(
                "Trying Playwright"
            )

            try:

                html = get_page_with_playwright(
                    current_url
                )

                soup = BeautifulSoup(
                    html,
                    "html.parser"
                )

                print(
                    "Playwright successful."
                )

            except Exception as e:

                print(
                    "Playwright error:",
                    e
                )

                continue

        else:

            print(
                "Requests content is usable"
            )


        # --------------------------------
        # Clean webpage text
        # --------------------------------

        text = clean_webpage(html)


        # --------------------------------
        # Extract title
        # --------------------------------

        title = (
            soup.title.get_text(
                strip=True
            )
            if soup.title
            else ""
        )


        # --------------------------------
        # Create LangChain Document
        # --------------------------------

        document = Document(
            page_content=text,
            metadata={
                "source": current_url,
                "title": title,
                "type": "website"
            }
        )

        documents.append(document)


        print(
            "Text characters:",
            len(text)
        )


        # --------------------------------
        # Extract links
        # --------------------------------

        links = soup.find_all("a")

        print(
            "Links found:",
            len(links)
        )


        for link in links:

            href = link.get("href")

            if not href:
                continue


            # Convert relative URL
            full_url = urljoin(
                current_url,
                href
            )


            # Normalize URL
            full_url = normalize_url(
                full_url
            )


            # Get domain
            link_domain = urlparse(
                full_url
            ).netloc


            # Only keep internal URLs
            if link_domain != base_domain:
                continue


            # Already visited?
            if full_url in visited:
                continue


            # Already waiting?
            if full_url in to_visit:
                continue


            # Add to queue
            to_visit.append(full_url)


    # --------------------------------
    # Return website Documents
    # --------------------------------

    print("\nCrawling finished!")

    print(
        "Total URLs found:",
        len(visited)
    )

    print(
        "Total documents:",
        len(documents)
    )

    return documents