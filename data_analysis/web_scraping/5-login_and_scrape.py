#!/usr/bin/env python3
"""Log into quotes.toscrape.com and scrape the quotes visible
once authenticated."""
import requests
from bs4 import BeautifulSoup


def login_and_scrape(login_url, user, pwd):
    """Log in with a CSRF-protected form, then scrape quotes.

    Args:
        login_url (str): The login page, e.g.
            "https://quotes.toscrape.com/login".
        user (str): Username to submit.
        pwd (str): Password to submit.

    Returns:
        list[dict]: One dict per quote, each shaped like:
            {"text": str, "author": str, "tags": list[str]}
    """
    # requests.Session() keeps cookies (like the session cookie the server
    # sets after login) around automatically between requests, the same
    # way a browser would. If we used plain requests.get()/post() calls
    # instead, each one would be a totally separate, cookie-less
    # connection, and the site would never "remember" that we logged in.
    session = requests.Session()

    # Step 1: GET the login page so we can read the CSRF token it embeds.
    # A CSRF token is a random, one-time value the server generates and
    # hides in the login form; the server only accepts a login POST if it
    # comes back with the *same* token it just handed out, as proof the
    # request really came from that form (and not some other site trying
    # to submit logins on your behalf).
    login_page = session.get(login_url)
    login_page.raise_for_status()

    soup = BeautifulSoup(login_page.text, "html.parser")

    # The token lives in a hidden field: <input name="csrf_token" value="...">
    # We find that <input> tag and read its "value" attribute.
    csrf_token = soup.find("input", attrs={"name": "csrf_token"})["value"]

    # Step 2: POST our credentials, along with that same CSRF token, back
    # to the login URL. This is exactly what happens when you click
    # "Submit" on a login form in a browser - the form's fields get sent
    # as a POST request body.
    payload = {
        "username": user,
        "password": pwd,
        "csrf_token": csrf_token,
    }
    response = session.post(login_url, data=payload)
    response.raise_for_status()

    # Step 3: now that the session cookie proves we're logged in, GET the
    # quotes page. Because we're reusing the same `session` object (not
    # plain `requests`), this request automatically carries the cookie
    # the server set during login.
    quotes_page = session.get("https://quotes.toscrape.com/")
    quotes_page.raise_for_status()

    soup = BeautifulSoup(quotes_page.text, "html.parser")

    # Same parsing logic as Task 1: each quote lives in a
    # <div class="quote">, with the text/author/tags nested inside it.
    quotes = []
    for quote_div in soup.find_all("div", class_="quote"):
        text = quote_div.find("span", class_="text").get_text()
        author = quote_div.find("small", class_="author").get_text()
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