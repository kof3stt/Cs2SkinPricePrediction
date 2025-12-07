import cloudscraper
from fake_useragent import FakeUserAgent
import json
import pandas as pd
from datetime import datetime, timezone
import matplotlib.pyplot as plt


class CsOnline:
    def __init__(
        self,
        base_url="https://api.tracker.gg/api/v1/gamepopulation/steam/730/entries",
        save_dir="cs_online.json",
        min_date=None,
        max_date=None,
    ):
        self.base_url = base_url
        self.save_dir = save_dir

        self.min_date = (
            datetime(year=2012, month=9, day=12, tzinfo=timezone.utc)
            if min_date is None
            else datetime.fromisoformat(min_date).replace(tzinfo=timezone.utc)
        )
        self.max_date = (
            datetime.now(tz=timezone.utc)
            if max_date is None
            else datetime.fromisoformat(max_date).replace(tzinfo=timezone.utc)
        )

        self.rangeStart = int(self.min_date.timestamp() * 1000)
        self.rangeEnd = int(self.max_date.timestamp() * 1000)

        self.params = {
            "platform": "all",
            "rangeEnd": self.rangeEnd,
            "rangeStart": self.rangeStart,
        }

        self.df = None

        self.scraper = cloudscraper.create_scraper()
        self.ua = FakeUserAgent()

    def run(self):
        response = self.scraper.get(
            self.base_url, params=self.params, headers={"User-Agent": self.ua.random}
        )
        print(f"url: {response.url}, status_code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()["data"]
            self.df = pd.DataFrame(data)
            self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])
            self.save_to_json(data)
        else:
            print(f"Ошибка при запросе, код: {response.status_code}")

    def save_to_json(self, data):
        with open(self.save_dir, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)

    def get_online_stats(self):
        self.df.info()

        min_avg_online = self.df["playerCount"].min()
        max_avg_online = self.df["playerCount"].max()

        min_row = self.df.loc[self.df["playerCount"] == min_avg_online].iloc[0]
        max_row = self.df.loc[self.df["playerCount"] == max_avg_online].iloc[0]

        print(
            f"Минимальное среднее число игроков за день: {min_avg_online}, зафиксировано: {min_row['timestamp']}"
        )
        print(
            f"Максимальное среднее число игроков за день: {max_avg_online}, зафиксировано: {max_row['timestamp']}"
        )

    def plot_data(self):
        plt.figure(figsize=(12, 6))
        plt.plot(
            self.df["timestamp"],
            self.df["playerCount"],
            label="Peak Players",
            color="blue",
        )
        plt.xlabel("Дата")
        plt.ylabel("Число игроков")
        plt.title("Counter-Strike 2: Онлайн")
        plt.legend()
        plt.grid(True)
        plt.show()


cs_online_parser = CsOnline()
cs_online_parser.run()
# cs_online_parser.get_online_stats()
cs_online_parser.plot_data()
