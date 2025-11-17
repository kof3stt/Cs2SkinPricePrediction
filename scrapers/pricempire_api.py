import json
import sys
import os
import time
import logging
from datetime import datetime
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy import create_engine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from database import PriceTimeSeries, Item, create_engine_with_config, create_tables

import cloudscraper


LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

log_file = os.path.join(LOG_DIR, "pricempire_api.log")

if os.path.exists(log_file):
    os.remove(log_file)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class PriceEmpireAPI:
    def __init__(self, items_json_path, cookies_file=None, delay = 4):
        self.base_url = "https://pricempire.com/api-data/v1/item/chart-providers"
        self.data = self.load_data(items_json_path)
        self.cookies = self.load_cookies(cookies_file) if cookies_file else {}
        self.scraper = cloudscraper.create_scraper()
        self.delay = delay

        engine = create_engine_with_config()
        Session = sessionmaker(bind=engine)
        self.session = Session()

        logger.info("Инициализация PriceEmpireAPI завершена")

    def load_data(self, path):
        with open(path, encoding="utf-8") as f:
            logger.info(f"Загрузка данных из {path}")
            return json.load(f)

    def load_cookies(self, cookies_file):
        try:
            with open(cookies_file, "r", encoding="utf-8") as f:
                cookies_str = f.read().strip()
            cookies = {}
            for c in cookies_str.split(";"):
                if "=" in c:
                    name, value = c.strip().split("=", 1)
                    cookies[name.strip()] = value.strip()
            logger.info(f"Файл cookie успешно загружен: {cookies_file}")
            return cookies
        except Exception as e:
            logger.error(f"Ошибка при загрузке cookies: {e}")
            return {}

    def get_item_price(
        self, pricempire_id, providers=["csfloat", "buff163", "steam"], days=10000
    ):
        headers = { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36', 'Accept': '*/*', 'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7', 'Accept-Encoding': 'gzip, deflate, br, zstd', 'Referer': 'https://pricempire.com/cs2/items/awp-dragon-lore-souvenir-factory-new', 'Origin': 'https://pricempire.com', 'Connection': 'keep-alive' }

        params = {
            "id": str(pricempire_id),
            "days": str(days),
            "providers": ",".join(providers),
        }

        max_retries = 5
        delay = self.delay

        for attempt in range(max_retries):
            response = self.scraper.get(
                self.base_url, params=params, cookies=self.cookies, headers=headers
            )

            if response.status_code == 200:
                try:
                    data = response.json()
                    # сбрасываем задержку, если всё успешно
                    self.delay = 5
                    return data
                except Exception as e:
                    logger.error(f"[{pricempire_id}] Ошибка парсинга JSON: {e}")
                    break  # не имеет смысла повторять при JSON-ошибке

            # ошибка — увеличиваем задержку
            logger.warning(f"[{pricempire_id}] HTTP {response.status_code}, повтор через {delay} сек.")
            time.sleep(delay)
            delay *= 2  # экспоненциальный рост задержки (5, 10, 20, 40, ...)

        logger.error(f"[{pricempire_id}] Не удалось получить данные после {max_retries} попыток.")
        return {}

    def save_time_series(self, item_id, provider, series):
        """Сохраняет список [timestamp, price, listings] в БД с защитой от дубликатов"""
        if not series:
            return

        values = [
            {
                "item_id": item_id,
                "provider": provider,
                "timestamp": datetime.fromtimestamp(ts),
                "price": price / 100,
                "listings": listings
            }
            for ts, price, listings in series
        ]

        stmt = insert(PriceTimeSeries).values(values)
        stmt = stmt.on_conflict_do_update(
            index_elements=["item_id", "provider", "timestamp"],
            set_={
                "price": stmt.excluded.price,
                "listings": stmt.excluded.listings,
            },
        )

        try:
            self.session.execute(stmt)
            self.session.commit()
            logger.info(f"✅ Сохранено {len(series)} точек для item_id={item_id}, provider={provider}")
        except Exception as e:
            self.session.rollback()
            logger.error(f"Ошибка при сохранении данных item_id={item_id}: {e}")

    def parse(self):
        """Перебирает все items и сохраняет временные ряды в БД"""
        logger.info("🚀 Начало парсинга данных с PriceEmpire")
        for item_data in self.data["items"]:
            pricempire_id = item_data.get("pricempire_id")
            if not pricempire_id:
                continue

            item = (
                self.session.query(Item).filter_by(pricempire_id=pricempire_id).first()
            )
            if not item:
                logger.warning(f"Нет данных для id={pricempire_id}")
                continue

            data = self.get_item_price(pricempire_id)
            if not data:
                print(f"Error parsing id {pricempire_id}")
                continue

            for provider, series in data.items():
                if series:
                    self.save_time_series(item.item_id, provider, series)
        logger.info("🏁 Парсинг завершен")


if __name__ == "__main__":
    api = PriceEmpireAPI("data/items_data.json", cookies_file="pricempire_cookies.txt")
    api.parse()
