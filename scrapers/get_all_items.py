import requests
import cloudscraper
import json
from fake_useragent import FakeUserAgent


def check_clash_gg_api():
    url = "https://inventory.clash.gg/api/GetItemsList/v2"

    try:
        response = cloudscraper.create_scraper().get(
            url,
            timeout=10,
            proxies={
                "http": "http://klikushin2004_gmail_com:333c8b012c@213.232.117.198:30013",
                "https": "http://klikushin2004_gmail_com:333c8b012c@213.232.117.198:30013",
            },
            headers={"User-Agent": FakeUserAgent().random},
        )
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            print(f"Ошибка доступа: {response.status_code}")
            return None
    except Exception as e:
        print(f"Ошибка подключения: {e}")
        return None


data = check_clash_gg_api()
if data is not None:
    with open("clash_items.json", "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)
