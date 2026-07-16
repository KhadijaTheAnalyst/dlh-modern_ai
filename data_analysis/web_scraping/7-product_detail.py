#!/usr/bin/env python3
"""Scrape a single product's detail page using Selenium (headless Chrome)."""
import time
from selenium import webdriver


def scrape_product_detail(url, delay=2.0):
    """Scrape one product's details from its detail page.

    Args:
        url (str): The product detail page, e.g.
            "https://webscraper.io/test-sites/e-commerce/static/product/32".
        delay (float): Seconds to wait after the page loads before reading
            it, to give slower-rendering content time to appear.

    Returns:
        dict: {"title": str, "price": str, "description": str, "rating": int}
    """
    options = webdriver.ChromeOptions()

    # Legacy headless flag - more stable than "--headless=new" on some
    # Chrome/driver version combinations in constrained container
    # environments.
    options.add_argument("--headless")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--no-sandbox")

    # Prevents Chrome's renderer process from crashing when /dev/shm is
    # too small, which is common in containerized/sandboxed environments.
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(options=options)

    try:
        driver.get(url)

        # Pause for the requested number of seconds before reading the
        # page. Unlike Task 6's fixed 1-second buffer, here the caller
        # controls exactly how long to wait via the `delay` parameter.
        time.sleep(delay)

        # A detail page's info block looks roughly like:
        #   <div class="caption">
        #     <h4 class="pull-right price">$306.99</h4>
        #     <h4><a class="title" title="Aspire E1-510">Aspire E1-510</a></h4>
        #     <p class="description">15.6", Pentium N3520 2.16GHz, ...</p>
        #   </div>
        # Both the price AND the title sit in separate <h4> tags inside
        # .caption, in that order - price first, title second - so we
        # grab BOTH <h4> elements as a list and pick them out by position.
        # The locator is passed as a plain string pair, matching what
        # the underlying element-finding API expects.
        caption_headings = driver.find_elements(
            "css selector", ".caption h4"
        )
        price = caption_headings[0].text
        # index 1 = the second <h4> = the one holding the title
        title = caption_headings[1].text

        description = driver.find_element(
            "css selector", ".description"
        ).text

        # The star rating on this site's detail page isn't a number
        # anywhere in the text - it's shown visually as one small star
        # icon element PER star. The selector below targets any element
        # (regardless of tag) that has "ws-icon-star" as an EXACT class
        # token - the [class~='...'] syntax matches only that whole
        # token, which correctly excludes a similarly-named but
        # different class like an empty/outline star variant.
        # So the rating is simply a COUNT of how many filled-star
        # elements exist, not a value we read off a single element.
        star_elements = driver.find_elements(
            "css selector", ".ratings [class~='ws-icon-star']"
        )
        rating = len(star_elements)

        return {
            "title": title,
            "price": price,
            "description": description,
            "rating": rating,
        }
    finally:
        driver.quit()
