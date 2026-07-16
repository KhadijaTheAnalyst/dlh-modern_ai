#!/usr/bin/env python3
"""Scrape a static product listing page using Selenium (headless Chrome)."""
import time
from selenium import webdriver


def scrape_products(url):
    """Scrape all products on a webscraper.io-style category page.

    Args:
        url (str): The product category page, e.g.
            "https://webscraper.io/test-sites/e-commerce/static/computers/laptops".

    Returns:
        list[dict]: One dict per product, each shaped like:
            {"title": str, "price": str, "description": str, "rating": int}
    """
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(options=options)

    try:
        driver.get(url)
        time.sleep(1)

        products = []
        product_cards = driver.find_elements("class name", "thumbnail")

        for card in product_cards:
            title_element = card.find_element("class name", "title")
            title = title_element.get_attribute("title")

            price = card.find_element("class name", "price").text

            description = card.find_element(
                "class name", "description"
            ).text

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
        driver.quit()
