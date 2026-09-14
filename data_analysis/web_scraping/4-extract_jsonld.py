#!/usr/bin/env python3
"""Extract quote data embedded as JSON-LD <script> blocks in a page."""
import json
from bs4 import BeautifulSoup
fetch_html = __import__('0-fetch_html').fetch_html


def _normalize_tags(keywords):
    """Turn a JSON-LD 'keywords' value into a clean list of tag strings.

    'keywords' can show up as:
      - a list already, e.g. ["fun", "test"]
      - a single comma-separated string, e.g. "fun, test"
      - missing entirely (None)
    """
    if keywords is None:
        return []
    if isinstance(keywords, list):
        return keywords
    # It's a string: split on commas and strip stray whitespace from
    # each piece, e.g. "fun, test" -> ["fun", " test"] -> ["fun", "test"]
    return [tag.strip() for tag in keywords.split(",")]


def extract_jsonld(url):
    """Extract quotes from JSON-LD <script> blocks on a page.

    Args:
        url (str): The Quotes List endpoint, e.g.
            "https://quotes.toscrape.com/".

    Returns:
        list[dict]: One dict per Quote object found, each shaped like:
            {"text": str, "author": str or None, "tags": list[str]}
    """
    html = fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")

    quotes = []

    # JSON-LD is structured data websites embed for search engines, e.g.:
    #   <script type="application/ld+json">
    #     { "@context": "...", "@type": "Quote", "text": "...", ... }
    #   </script>
    # find_all() here filters by the "type" attribute rather than a class,
    # since that's how these script tags identify themselves.
    scripts = soup.find_all("script", type="application/ld+json")

    for script in scripts:
        # .string gives the raw text content sitting between the
        # <script> and </script> tags - the JSON-LD text itself.
        try:
            payload = json.loads(script.string)
        except (json.JSONDecodeError, TypeError):
            # Skip any script block that isn't valid JSON rather than
            # letting one bad block crash the whole function.
            continue

        # A single <script> block might hold ONE JSON-LD object, or a
        # LIST of them. Normalize both cases into a list so we can loop
        # over it the same way either way.
        entries = payload if isinstance(payload, list) else [payload]

        for entry in entries:
            # Only entries describing a Quote are relevant here; a page
            # could embed other JSON-LD types (Organization, WebSite, etc.)
            # that we want to ignore.
            if entry.get("@type") != "Quote":
                continue

            quotes.append({
                "text": entry.get("text"),
                # "author" is itself a nested object, e.g.
                # {"@type": "Person", "name": "Tester"} - we only want
                # the name. .get("author", {}) falls back to an empty
                # dict if "author" is missing entirely, so the second
                # .get("name") doesn't blow up with an AttributeError.
                "author": entry.get("author", {}).get("name"),
                "tags": _normalize_tags(entry.get("keywords")),
            })

    return quotes
