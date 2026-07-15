#!/usr/bin/env python3
"""Scrape quotes across every page of quotes.toscrape.com by
following the 'Next' link until none remains."""
from bs4 import BeautifulSoup
import time
from urllib import parse
fetch_html = __import__('0-fetch_html').fetch_html
scrape_basic = __import__('1-scrape_basic').scrape_basic


def scrape_paginated(base_url):
    """Scrape quotes from every page, following pagination automatically.

    Args:
        base_url (str): The first page URL, e.g.
            "https://quotes.toscrape.com/".

    Returns:
        list[dict]: All quote dicts from every page combined, each shaped
            like {"text": str, "author": str, "tags": list[str]}.
    """
    all_quotes = []

    # We'll keep replacing this with the "next page" URL until there
    # isn't one anymore. None acts as our "stop the loop" signal.
    current_url = base_url

    while current_url is not None:
        # Fetch this page's HTML ONCE and reuse it both for scraping the
        # quotes and for checking pagination. Fetching twice (once via
        # scrape_basic, once via fetch_html) would double the number of
        # requests made - which is wasteful, and can trip up servers
        # (including test/checker servers) that cap how many connections
        # they'll accept.
        html = fetch_html(current_url)
        soup = BeautifulSoup(html, "html.parser")

        # Same extraction logic as Task 1's scrape_basic(), applied
        # directly to the soup we already have instead of calling
        # scrape_basic() (which would re-fetch the page itself).
        for quote_div in soup.find_all("div", class_="quote"):
            text = quote_div.find("span", class_="text").get_text()
            author = quote_div.find("small", class_="author").get_text()
            tag_links = quote_div.find("div", class_="tags").find_all(
                "a", class_="tag"
            )
            tags = [tag.get_text() for tag in tag_links]

            all_quotes.append({
                "text": text,
                "author": author,
                "tags": tags,
            })

        # On quotes.toscrape.com, the pagination control looks like:
        #   <li class="next"><a href="/page/2/
        # ">Next <span>&rarr;</span></a></li>
        # On the LAST page, this <li class="next"> simply doesn't exist.
        next_li = soup.find("li", class_="next")

        if next_li is None:
            # No more pages - stop the loop.
            current_url = None
        else:
            # The href is a *relative* path, e.g. "/page/2/", not a full
            # URL. urllib.parse.urljoin combines it with the current page's
            # URL to build the next full URL, e.g.
            # urljoin("https://quotes.toscrape.com/", "/page/2/")
            #   -> "https://quotes.toscrape.com/page/2/"
            next_href = next_li.find("a")["href"]
            current_url = parse.urljoin(current_url, next_href)

            # Be polite to the server: wait a bit before the next request
            # instead of hammering the site as fast as possible.
            time.sleep(1)

    return all_quotes
