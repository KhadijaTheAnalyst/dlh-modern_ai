#!/usr/bin/env python3
"""Scrape an infinite-scroll product page using Selenium (headless Chrome),
scrolling repeatedly to load every product before extracting."""
import time
from selenium import webdriver


def scroll_and_scrape(url, scroll_pause=2.0):
    """Scroll a JS infinite-scroll page to the bottom repeatedly, then
    scrape every unique product that got loaded.

    Args:
        url (str): The infinite-scroll product page.
        scroll_pause (float): Maximum seconds to wait after each scroll
            for new products to finish loading in. Waiting stops early
            as soon as new content is detected.

    Returns:
        list[dict]: One dict per unique product.
    """
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(options=options)

    try:
        driver.get(url)

        # execute_script() lets us run arbitrary JavaScript inside the
        # loaded page, as if we'd typed it into the browser's console.
        # "return document.body.scrollHeight" reports how tall the page
        # currently is, in pixels. As more products load in below, this
        # number grows.
        last_height = driver.execute_script(
            "return document.body.scrollHeight"
        )

        # How often to re-check the page height while waiting for new
        # content, in seconds. Checking frequently lets us move on as
        # soon as new products actually appear, instead of always
        # waiting the full scroll_pause even when content loads faster -
        # this matters a lot when many small batches need to load one
        # after another under a time budget.
        poll_interval = 0.2

        while True:
            # Scroll the browser window all the way to the current
            # bottom of the page. On an infinite-scroll site, reaching
            # the bottom is exactly what triggers the page's JavaScript
            # to fetch and append the next batch of products.
            driver.execute_script(
                "window.scrollTo(0, document.body.scrollHeight);"
            )

            # Poll for a height change instead of always sleeping the
            # full scroll_pause. scroll_pause becomes the MAXIMUM wait
            # per batch, not a fixed one - as soon as new content has
            # appended, we stop waiting and move on immediately.
            elapsed = 0.0
            new_height = last_height
            while elapsed < scroll_pause:
                time.sleep(poll_interval)
                elapsed += poll_interval
                new_height = driver.execute_script(
                    "return document.body.scrollHeight"
                )
                if new_height != last_height:
                    break

            # If the height never changed after the full scroll_pause
            # window, no new content was loaded - we've hit the true
            # bottom of the page and every product is now present.
            # Stop scrolling.
            if new_height == last_height:
                break

            last_height = new_height

        products = []
        # This tracks which (title, price) combos we've already added,
        # so we don't add the same product twice. A set gives fast
        # "have I seen this before?" lookups.
        seen = set()

        product_cards = driver.find_elements("class name", "thumbnail")

        for card in product_cards:
            title = card.find_element(
                "class name", "title"
            ).get_attribute("title")
            price = card.find_element("class name", "price").text
            description = card.find_element(
                "class name", "description"
            ).text

            star_elements = card.find_elements(
                "css selector", ".ratings [class~='ws-icon-star']"
            )
            rating = len(star_elements)

            # A tuple like ("Aspire E1-510", "$306.99") is hashable, so
            # it can go straight into a set - unlike a list or dict,
            # which couldn't be used as a set member.
            key = (title, price)
            if key in seen:
                # Already recorded this exact product - skip it so it
                # doesn't appear twice in the results.
                continue
            seen.add(key)

            products.append({
                "title": title,
                "price": price,
                "description": description,
                "rating": rating,
            })

        return products
    finally:
        driver.quit()
