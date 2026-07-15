#!/usr/bin/env python3
"""Scrape a static product listing page using Selenium (headless Chrome)."""
import time
from selenium import webdriver


def scrape_products_list(url):
    """Scrape all products on a webscraper.io-style category page.

    Args:
        url (str): The product category page, e.g.
            "https://webscraper.io/test-sites/e-commerce/static/computers/laptops".

    Returns:
        list[dict]: One dict per product, each shaped like:
            {"title": str, "price": str, "description": str, "rating": int}
    """
    # Selenium options configure HOW the browser launches, before we
    # even open a page.
    options = webdriver.ChromeOptions()

    # "--headless=new" runs Chrome with no visible UI window - useful on
    # servers/Colab where there's no display to show a real browser on.
    options.add_argument("--headless=new")

    # Set a fixed browser window size. Some pages render their layout
    # differently (or hide elements) at very small/default sizes, so we
    # pin it to a common desktop resolution for consistent results.
    options.add_argument("--window-size=1920,1080")

    # "--no-sandbox" disables Chrome's own process sandboxing. It's
    # commonly required when running Chrome as root inside containers
    # (like Colab or CI environments), where the sandbox setup Chrome
    # expects isn't available.
    options.add_argument("--no-sandbox")

    # This actually launches a real (headless) Chrome browser process,
    # configured with the options above.
    driver = webdriver.Chrome(options=options)

    try:
        # Navigate the browser to the target page, just like typing a
        # URL into the address bar and hitting Enter.
        driver.get(url)

        # A short pause to let the page finish rendering before we start
        # searching it for elements. This is a static page so it likely
        # doesn't need this, but it's cheap insurance against timing
        # issues (e.g. slow network, slow initial paint).
        time.sleep(1)

        products = []

        # Every product card on this site is a <div class="thumbnail">.
        # find_elements() (plural) returns a list of ALL matches, unlike
        # find_element() (singular) which returns just the first one.
        # We use the raw locator string "class name" instead of
        # importing selenium's By class, since By.CLASS_NAME is just
        # that same string under the hood - this keeps us within the
        # "only import time and webdriver" constraint.
        product_cards = driver.find_elements("class name", "thumbnail")

        for card in product_cards:
            # The product name lives on an <a class="title"> tag, e.g.
            #   <a href="..." class="title"
            #           title="Packard 255 G2">Packard 255 G2</a>
            # We read the "title" ATTRIBUTE (not the visible text), since
            # the visible text can sometimes get truncated with "..." for
            # long names, but the title attribute always holds the full name.
            title_element = card.find_element("class name", "title")
            title = title_element.get_attribute("title")

            # The price sits in <h4 class="price">$416.99</h4>.
            # .text reads the visible text content of the element.
            price = card.find_element("class name", "price").text

            # The description sits in <p class="description">...</p>.
            description = card.find_element(
                "class name", "description"
            ).text

            # The star rating doesn't have its own class - it's a <p> tag
            # with a "data-rating" attribute holding the numeric value,
            # nested inside a <div class="ratings"> wrapper, e.g.
            #   <div class="ratings">
            #     <p data-rating="2">...</p>
            #     ...
            #   </div>
            # A CSS selector lets us target "any <p> that has a
            # data-rating attribute" inside the ratings div.
            rating_element = card.find_element(
                "css selector", ".ratings p[data-rating]"
            )
            rating = int(rating_element.get_attribute("data-rating"))

            products.append({
                "title": title,
                "price": price,
                "description": description,
                "rating": rating,
            })

        return products
    finally:
        # Always close the browser process when we're done, whether we
        # succeeded or an exception was raised above. Leaving Chrome
        # processes running would waste memory/resources over time.
        driver.quit()
