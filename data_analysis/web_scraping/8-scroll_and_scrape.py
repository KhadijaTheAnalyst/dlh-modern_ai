#!/usr/bin/env python3
"""Scrape an infinite-scroll product page using Selenium (headless Chrome),
scrolling repeatedly to load every product before extracting."""
import time
from selenium import webdriver
from selenium.webdriver.common.by import By


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
    options.add_argument("--headless=new")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--no-sandbox")

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

        product_cards = driver.find_elements(By.CLASS_NAME, "thumbnail")

        for card in product_cards:
            title = card.find_element(
                By.CLASS_NAME, "title"
            ).get_attribute("title")
            price = card.find_element(By.CLASS_NAME, "price").text
            description = card.find_element(
                By.CLASS_NAME, "description"
            ).text

            star_elements = card.find_elements(
                By.CSS_SELECTOR, ".ratings p.ws-icon.ws-icon-star"
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
