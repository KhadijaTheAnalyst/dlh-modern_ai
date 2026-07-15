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
        # Step 1: scrape the quotes on this page. scrape_basic() does its
        # own fetch_html() call internally and returns a list of dicts -
        # exactly like Task 1, just for whichever page we're on right now.
        page_quotes = scrape_basic(current_url)
        all_quotes.extend(page_quotes)

        # Step 2: fetch the raw HTML again ourselves (separately from
        # scrape_basic) so we can look for the "Next" link in it.
        html = fetch_html(current_url)
        soup = BeautifulSoup(html, "html.parser")

        # On quotes.toscrape.com, the pagination control looks like:
        #   <li class="next"><a href="/page/2/">Next <span>&rarr;</span></a></li>
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
