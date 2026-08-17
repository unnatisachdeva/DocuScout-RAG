#this file has one job : URL → browser → rendered HTML.

from playwright.sync_api import sync_playwright

def get_page_with_playwright(url):  
    with sync_playwright() as p:

        browser = p.chromium.launch(
            headless=True
        )

        page = browser.new_page()

        page.goto(url, wait_until="networkidle", timeout=10000)

        html = page.content()

        browser.close()

        return html