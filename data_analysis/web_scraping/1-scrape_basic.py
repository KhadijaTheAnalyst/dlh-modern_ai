#!/usr/bin/env python3
"""Scrape quote data from a single page of quotes.toscrape.com."""
from bs4 import BeautifulSoup
fetch_html = __import__('0-fetch_html').fetch_html


def scrape_basic(url):
    """Scrape the quotes on a single quotes.toscrape.com page.

    Args:
        url (str): The Quotes List endpoint, e.g.
            "https://quotes.toscrape.com/".

    Returns:
        list[dict]: One dict per quote, each shaped like:
            {"text": str, "author": str, "tags": list[str]}
    """
    # Step 1: download the raw HTML for the page (reusing task 0's helper).
    html = fetch_html(url)

    # Step 2: hand the HTML string to BeautifulSoup so we can navigate it
    # like a tree instead of parsing raw text ourselves. "html.parser" is
    # Python's built-in parser, so no extra dependency (like lxml) is needed.
    soup = BeautifulSoup(html, "html.parser")

    quotes = []

    # On quotes.toscrape.com, every quote lives inside a
    # <div class="quote"> block. find_all() grabs every one of them
    # on the page, in the order they appear.
    for quote_div in soup.find_all("div", class_="quote"):
        # The quote text sits in <span class="text">"..."</span>.
        # .get_text() pulls out just the visible text, without the
        # surrounding HTML tags.
        text = quote_div.find("span", class_="text").get_text()

        # The author's name sits in <small class="author">Name</small>.
        author = quote_div.find("small", class_="author").get_text()

        # Tags are a list of <a class="tag">tag-name</a> elements inside
        # <div class="tags">. We collect each one's text into a list.
        tag_links = quote_div.find("div", class_="tags").find_all(
            "a", class_="tag"
        )
        tags = [tag.get_text() for tag in tag_links]

        quotes.append({
            "text": text,
            "author": author,
            "tags": tags,
        })

    return quotes
