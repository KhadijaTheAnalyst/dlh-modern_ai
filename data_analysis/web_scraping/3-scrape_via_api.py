#!/usr/bin/env python3
"""Scrape quote data directly from quotes.toscrape.com's JSON API,
paging through until the last page is reached."""
import json
fetch_html = __import__('0-fetch_html').fetch_html

# A browser-like User-Agent. Requests sent without one (the default for
# python-requests) can get rate-limited or refused by the server partway
# through a run of several rapid consecutive requests.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/115.0.0.0 Safari/537.36"
    )
}


def scrape_via_api(base_url):
    """Fetch every quote from the site's JSON API endpoints.

    Args:
        base_url (str): The root URL of the site, e.g.
            "https://quotes.toscrape.com" (no trailing slash needed).

    Returns:
        list[dict]: One dict per quote, each shaped like:
            {"text": str, "author": str, "tags": list[str]}
    """
    all_quotes = []
    page = 1

    while True:
        # Build this page's API URL, e.g.
        # "https://quotes.toscrape.com/api/quotes?page=1", then "...page=2"
        # and so on.
        url = f"{base_url}/api/quotes?page={page}"

        # Even though this endpoint returns JSON (not HTML), fetch_html()
        # still works fine here - it just does a GET request and returns
        # the raw response body as a string. We just have to parse that
        # string as JSON ourselves.
        raw_response = fetch_html(url, headers=_HEADERS)

        # json.loads() turns the JSON text into native Python data:
        # objects -> dict, arrays -> list, strings/numbers stay the same.
        data = json.loads(raw_response)

        # The payload looks like:
        # {
        #   "has_next": true/false,
        #   "page": <int>,
        #   "quotes": [ {"text": ..., "author": {"name": ...}, "tags": [...]}, ... ]
        # }
        for quote in data["quotes"]:
            all_quotes.append({
                "text": quote["text"],
                # "author" in the API response is itself a nested dict
                # (with name, slug, goodreads_link) - we only want the name.
                "author": quote["author"]["name"],
                "tags": quote["tags"],
            })

        # "has_next" tells us directly whether there's another page to
        # fetch - no need to guess or look for a "Next" link like in
        # the HTML version (Task 2).
        if not data["has_next"]:
            break

        page += 1

    return all_quotes
