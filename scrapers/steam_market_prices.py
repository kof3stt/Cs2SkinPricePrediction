import json
import cloudscraper
from bs4 import BeautifulSoup


class PriceEmpireParser:
    def __init__(self, domain, base_url, params, max_retries=25, proxies={}):
        self.domain = domain
        self.base_url = base_url
        self.params = params
        self.max_retries = max_retries
        self.proxies = None if not proxies else proxies

        self.items_links = []

    def get_items(self):
        response = cloudscraper.create_scraper().get(
            self.base_url, params=self.params, proxies=self.proxies
        )
        soup = BeautifulSoup(response.text, "lxml")

        current_page = 1

        while True:
            print(f"Parsing page: {response.url}")
            items = soup.select("div.results-container div.grid a")
            if not items:
                print(f"No items found on the page: {response.url}")
                break
            for item in items:
                item_page = self.domain + item["href"]
                self.items_links.append(item_page)

            print(f"\033[92mParsing OK: {response.url}\033[0m")

            current_page += 1

            for i in range(self.max_retries):
                self.params["page"] = current_page
                try:
                    response = cloudscraper.create_scraper().get(
                        self.base_url, params=self.params, proxies=self.proxies
                    )
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, "lxml")
                        success = True
                        break
                    else:
                        print(
                            f"\033[91mError parsing page {response.url}, attempt {i+1}/{self.max_retries}, status code: {response.status_code}\033[0m"
                        )
                except Exception as e:
                    print(
                        f"\033[91mError parsing page {response.url}, attempt {i+1}/{self.max_retries}, exception: {str(e)}\033[0m"
                    )

            if not success:
                print(f"Failed to get page {response.url} after {self.max_retries} attempts. Stopping.")
                break

    def save_item_links(self, path_file):
        with open(path_file, "w", encoding="utf-8") as file:
            json.dump(self.items_links, file, ensure_ascii=False, indent=4)


params = {"sort": "price-asc", "grouped": "false", "limit": "96"}
proxies = {"http": "http://5.79.73.131:13150", "https": "http://5.79.73.131:13150"}

price_empire = PriceEmpireParser(
    "https://pricempire.com",
    "https://pricempire.com/cs2-skin-search",
    params,
    proxies=proxies,
)

price_empire.get_items()
price_empire.save_item_links("pricempire_items_urls.json")
