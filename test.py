from playwright.sync_api import sync_playwright
import json

def scrape_playwright_docs():
    with sync_playwright() as p:

        # Launch browser
        browser = p.chromium.launch(headless=False)

        # Create page
        page = browser.new_page()

        # Open website
        page.goto("https://playwright.dev/", timeout=60000)

        # Wait for page load
        page.wait_for_load_state("networkidle")

        # Get page title
        title = page.title()

        print(f"Website Title: {title}")

        # -----------------------------
        # Scrape navigation links
        # -----------------------------
        nav_links = []

        nav_elements = page.locator("nav a").all()

        for nav in nav_elements:
            text = nav.inner_text().strip()
            href = nav.get_attribute("href")

            if text:
                nav_links.append({
                    "text": text,
                    "link": href
                })

        # -----------------------------
        # Scrape headings
        # -----------------------------
        headings = []

        heading_elements = page.locator("h1, h2, h3").all()

        for heading in heading_elements:
            text = heading.inner_text().strip()

            if text:
                headings.append(text)

        # -----------------------------
        # Scrape paragraphs
        # -----------------------------
        paragraphs = []

        paragraph_elements = page.locator("p").all()

        for para in paragraph_elements:
            text = para.inner_text().strip()

            if text:
                paragraphs.append(text)

        # -----------------------------
        # Store data
        # -----------------------------
        scraped_data = {
            "title": title,
            "navigation_links": nav_links,
            "headings": headings,
            "paragraphs": paragraphs
        }

        # Save to JSON
        with open("playwright_data.json", "w", encoding="utf-8") as f:
            json.dump(scraped_data, f, indent=4, ensure_ascii=False)

        print("\nData saved to playwright_data.json")

        browser.close()


if __name__ == "__main__":
    scrape_playwright_docs()