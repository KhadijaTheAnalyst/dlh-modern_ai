#!/usr/bin/env python3
"""Fetch the HTML content of a web page."""
import requests


def fetch_html(url, headers=None, timeout=10):
    """Fetch a web page and return its HTML as text.

    Args:
        url (str): The page to retrieve.
        headers (dict, optional): HTTP headers to send with the request.
        timeout (int, optional): Seconds to wait before aborting. Default 10.

    Returns:
        str: The full HTML of the response.

    Raises:
        requests.exceptions.HTTPError: If the response status is >= 400.
    """
    # Send the GET request.
    # - headers=None is fine: requests just falls back to its own defaults.
    #   Passing a real browser User-Agent here helps get past sites that
    #   block/limit requests that look like they're coming from a bot.
    # - timeout stops the request from hanging forever if the server
    #   never responds.
    response = requests.get(url, headers=headers, timeout=timeout)

    # requests does NOT raise an error on its own for 4xx/5xx status codes -
    # a 404 or 403 is still a "successful" HTTP exchange as far as the
    # connection is concerned. raise_for_status() is what turns a bad
    # status code (>= 400) into an actual Python exception (HTTPError),
    # e.g. "404 Client Error: Not Found for url: ...".
    # If the status is 2xx/3xx, this line does nothing and we move on.
    response.raise_for_status()

    # response.text decodes the raw bytes into a string using the encoding
    # requests detects from the response headers (or guesses otherwise).
    # This is the actual HTML source of the page.
    return response.text
