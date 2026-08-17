from bs4 import BeautifulSoup

def clean_webpage(html):

    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    # Remove elements that don't normally contain
    # useful content for RAG
    for element in soup([
        "script",
        "style",
        "noscript",
        "svg",
        "nav",
        "header",
        "footer"
    ]):
        element.decompose()


    # --------------------------------
    # Try <main> elements first
    # --------------------------------

    main_elements = soup.find_all("main")

    if main_elements:

        # Choose the main element containing
        # the most text
        main = max(
            main_elements,
            key=lambda element: len(
                element.get_text(
                    " ",
                    strip=True
                )
            )
        )

        text = main.get_text(
            " ",
            strip=True
        )

    else:

        # --------------------------------
        # Fallback: try <article>
        # --------------------------------

        article_elements = soup.find_all(
            "article"
        )

        if article_elements:

            article = max(
                article_elements,
                key=lambda element: len(
                    element.get_text(
                        " ",
                        strip=True
                    )
                )
            )

            text = article.get_text(
                " ",
                strip=True
            )

        else:

            # --------------------------------
            # Final fallback: entire page
            # --------------------------------

            text = soup.get_text(
                " ",
                strip=True
            )


    # Normalize whitespace
    text = " ".join(text.split())

    return text