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
        scroll_pause (float): Seconds to wait after each scroll for new
            products to finish loading in.

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

        last_height = driver.execute_script(
            "return document.body.scrollHeight"
        )

        while True:
            driver.execute_script(
                "window.scrollTo(0, document.body.scrollHeight);"
            )
            time.sleep(scroll_pause)

            new_height = driver.execute_script(
                "return document.body.scrollHeight"
            )

            if new_height == last_height:
                break

            last_height = new_height

        products = []
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

            key = (title, price)
            if key in seen:
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
