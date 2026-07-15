# Web Scraping Project

A progressive series of Python web-scraping exercises, moving from basic
static HTML parsing to session-based logins and JavaScript-rendered pages
with Selenium.

Target sites used throughout:
- [quotes.toscrape.com](https://quotes.toscrape.com/) — a scraping sandbox
  with a paginated quote list, a login form, a JSON API, and JSON-LD data.
- [webscraper.io test sites](https://webscraper.io/test-sites) — static,
  detail-page, and infinite-scroll e-commerce sandboxes.

## Requirements

```bash
pip install requests beautifulsoup4 selenium
```

Tasks 6–8 additionally require a Chrome/Chromium browser and a matching
`chromedriver` available on the system (Selenium 4.6+ can auto-manage the
driver; on Colab you may need to install Chromium manually — see notes
below).

## Files

| File | Task | Function |
|---|---|---|
| `0-fetch_html.py` | 0. Fetch HTML | `fetch_html(url, headers=None, timeout=10)` |
| `1-scrape_basic.py` | 1. Basic Static Scraping | `scrape_basic(url)` |
| `2-scrape_paginated.py` | 2. Pagination Handling | `scrape_paginated(base_url)` |
| `3-scrape_via_api.py` | 3. API-Based Scraping | `scrape_via_api(base_url)` |
| `4-extract_jsonld.py` | 4. JSON-LD Extraction | `extract_jsonld(url)` |
| `5-login_and_scrape.py` | 5. Login & Scrape | `login_and_scrape(login_url, user, pwd)` |
| `6-products_list.py` | 6. Scrape Static Products | `scrape_products_list(url)` |
| `7-product_detail.py` | 7. Scrape Single Product Detail | `scrape_product_detail(url, delay=2.0)` |
| `8-scroll_and_scrape.py` | 8. Scroll & Scrape Products | `scroll_and_scrape(url, scroll_pause=2.0)` |

## Task Overview

### 0. Fetch HTML
Thin wrapper around `requests.get()` that returns the raw HTML of a page as
a string, raising an exception on any HTTP status ≥ 400 via
`raise_for_status()`.

### 1. Basic Static Scraping
Fetches the quotes homepage and parses it with BeautifulSoup, pulling
`text`, `author`, and `tags` out of every `div.quote` block on the page.

### 2. Pagination Handling
Wraps Task 1's scraper in a loop that follows each page's `<li class="next">`
link (resolved to a full URL with `urllib.parse.urljoin`) until no next
page remains, combining every page's quotes into one list. Includes a
`time.sleep(1)` delay between page requests.

### 3. API-Based Scraping
Bypasses HTML entirely and pages through the site's JSON API
(`/api/quotes?page=N`), reading the `has_next` flag to know when to stop.

### 4. JSON-LD Extraction
Parses `<script type="application/ld+json">` blocks embedded in the page,
filtering for `"@type": "Quote"` entries and normalizing the `keywords`
field (which may arrive as either a list or a comma-separated string) into
a clean tag list.

### 5. Login & Scrape
Uses `requests.Session()` to persist cookies across requests: fetches the
login page to extract a hidden CSRF token, POSTs credentials + token back,
then scrapes the quotes page using the now-authenticated session.

### 6. Scrape Static Products
First Selenium-based task. Launches headless Chrome (1920×1080, sandbox
disabled) and scrapes every `div.thumbnail` product card on a static
e-commerce category page for `title`, `price`, `description`, and `rating`.

### 7. Scrape Single Product Detail
Same Selenium setup as Task 6, but for a single product's detail page.
Waits a configurable `delay` after page load, then reads title/price from
two `<h4>` tags inside `.caption` and derives the star rating by *counting*
filled star icon elements rather than reading a numeric attribute.

### 8. Scroll & Scrape Products
Handles a JavaScript infinite-scroll page: repeatedly scrolls to the
bottom via `execute_script("window.scrollTo(...)")`, comparing
`document.body.scrollHeight` before/after each scroll to detect when no
more products are loading. Deduplicates products by `(title, price)` pairs
before returning the final list.

## Usage

Each file is self-contained and callable via the numeric-prefixed import
pattern (required since filenames starting with a digit aren't valid
Python identifiers):

```python
fetch_html = __import__('0-fetch_html').fetch_html
scrape_basic = __import__('1-scrape_basic').scrape_basic

quotes = scrape_basic("https://quotes.toscrape.com/")
```

Or run the corresponding `N-main.py` test script directly:

```bash
./1-main.py
```

## Notes on Selenium (Tasks 6–8)

- All three tasks configure Chrome identically:
  ```python
  options.add_argument("--headless=new")
  options.add_argument("--window-size=1920,1080")
  options.add_argument("--no-sandbox")
  ```
- `driver.quit()` is always called in a `finally` block so the browser
  process is cleaned up even if scraping raises an exception.
- **On Google Colab**, Chrome/Chromedriver aren't preinstalled. Run this
  first:
  ```python
  !apt-get update -qq
  !apt-get install -qq chromium-chromedriver
  !pip install selenium --quiet
  ```
  If `webdriver.Chrome(options=options)` can't find the browser binary,
  set it explicitly: `options.binary_location = "/usr/bin/chromium-browser"`.

## Design Choices Worth Knowing

- **No regex** was used anywhere (per task constraints) — all extraction
  relies on BeautifulSoup's `find`/`find_all`, Selenium's
  `find_element(s)`/CSS selectors, or direct JSON parsing.
- **Copyright/robustness**: `fetch_html()` centralizes error handling
  (`raise_for_status()`) so every downstream scraper automatically fails
  loudly on 4xx/5xx instead of silently parsing an error page.
- **Politeness**: Task 2 sleeps between page requests; Tasks 6–8 use
  headless Chrome responsibly with fixed viewport sizes for consistent,
  reproducible element positions.