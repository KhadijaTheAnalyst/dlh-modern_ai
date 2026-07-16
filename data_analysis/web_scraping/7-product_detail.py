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
    options.add_argument("--headless")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(options=options)

    try:
        driver.get(url)
        time.sleep(delay)

        caption_headings = driver.find_elements(
            "css selector", ".caption h4"
        )
        price = caption_headings[0].text
        title = caption_headings[1].text

        description = driver.find_element(
            "css selector", ".description"
        ).text

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
