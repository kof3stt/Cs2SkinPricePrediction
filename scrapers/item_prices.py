import aiohttp
import asyncio
import json
import random
from fake_useragent import FakeUserAgent


class ItemPrices:
    def __init__(
        self,
        base_url="https://steamcommunity.com/market/search/render/",
        appid=730,
        proxies=None,
        max_connections=5,
    ):
        self.base_url = base_url
        self.appid = appid
        self.proxies = proxies or []
        self.semaphore = asyncio.Semaphore(max_connections)
        self.ua = FakeUserAgent()

    async def fetch_page(self, session, start, count):
        """Асинхронно получает одну страницу с товарами."""
        params = {"appid": self.appid, "norender": 1, "count": count, "start": start}

        delay = 2

        while True:
            try:
                proxy = random.choice(self.proxies) if self.proxies else None
                headers = {"User-Agent": self.ua.random}
                async with self.semaphore:
                    async with session.get(
                        self.base_url,
                        params=params,
                        proxy=proxy,
                        headers=headers,
                        timeout=5,
                    ) as response:
                        if response.status == 200:
                            print(
                                "\033[92m"
                                + f"PARSING OK, url: {response.url}"
                                + "\033[0m"
                            )
                            data = await response.json()
                            delay = 2
                            return data
                        else:
                            print(
                                "\033[91m"
                                + f"PARSING ERR ({response.status}), url: {response.url}"
                                + "\033[0m"
                            )
            except Exception as e:
                print("\033[91m" + f"Ошибка соединения: {e}" + "\033[0m")

            print(f"⏳ Повтор через {round(delay, 2)} сек...")
            await asyncio.sleep(delay)
            delay = min(delay * 2, 120)

    async def get_all_items(self):
        start = 0
        count = 10
        all_items = []

        async with aiohttp.ClientSession() as session:
            while True:
                data = await self.fetch_page(session, start, count)
                if not data:
                    break

                results = data.get("results", [])
                if not results:
                    break

                for item in results:
                    item_info = {
                        "name": item.get("name"),
                        "hash_name": item.get("hash_name"),
                        "sell_listings": item.get("sell_listings"),
                        "sell_price": item.get("sell_price"),
                        "sell_price_text": item.get("sell_price_text"),
                        "app_icon": item.get("app_icon"),
                        "app_name": item.get("app_name"),
                        "asset_description": item.get("asset_description"),
                        "sale_price_text": item.get("sale_price_text"),
                    }
                    all_items.append(item_info)

                start += count
                total = data.get("total_count", start)
                print(f"Загружено {len(all_items)} из {total}")

                if start >= total:
                    break

        self.save_to_all_items_json("all_items_async.json", all_items)
        return all_items

    def save_to_all_items_json(self, filename, data):
        with open(filename, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)


async def main():
    item_prices = ItemPrices(proxies=["http://5.79.73.131:13150"], max_connections=2)
    await item_prices.get_all_items()


if __name__ == "__main__":
    asyncio.run(main())
