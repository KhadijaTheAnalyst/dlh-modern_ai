#!/usr/bin/env python3
scrape_product_detail = __import__(
    '7-product_detail').scrape_product_detail
url = "https://webscraper.io/test-sites/e-commerce/static/product/32"
detail = scrape_product_detail(url)
print(f"Product details: {detail}")
$ ./7-main.py
Product details: {'title': 'Aspire E1-510', 'price': '$306.99', 'description': '15.6", Pentium N3520 2.16GHz, 4GB, 500GB, Linux', 'rating': 3}