from pytrends.request import TrendReq
from fake_useragent import FakeUserAgent
import pandas as pd
import matplotlib.pyplot as plt
import time
import os


class GoogelTrends:
    def __init__(self, max_retries=100, backoff_factor=0.2, proxies=""):
        self.max_retries = max_retries
        self.proxies = "" if not proxies else proxies
        self.ua = FakeUserAgent()

        requests_args = {
            "headers": {
                "User-Agent": self.ua.random
            }
        }

        # if proxies:
        #     requests_args['proxies'] = proxies

        self.pytrends = TrendReq(
            hl="en-US",
            tz=360,
            backoff_factor=backoff_factor,
            proxies=proxies,
        )

    def get_trends(self, keywords, timeframe="all", geo=""):
        all_trends = pd.DataFrame()

        for keyword in keywords:
            print(f"Fetching trend data for keyword: {keyword}")

            wait_time = 10
            retries = 0

            while retries < self.max_retries:
                try:
                    self.pytrends.build_payload([keyword], timeframe=timeframe, geo=geo)
                    data = self.pytrends.interest_over_time()

                    if data.empty:
                        print(f"No trend data found for the keyword: {keyword}")
                        break

                    data = data.drop(columns=["isPartial"])

                    if all_trends.empty:
                        all_trends = data.rename(columns={keyword: keyword})
                    else:
                        all_trends = all_trends.join(
                            data.rename(columns={keyword: keyword}), how="outer"
                        )

                    break

                except Exception as e:
                    retries += 1
                    print(f"Error fetching data for {keyword}: {e}")

                    if retries > self.max_retries:
                        print(f"Maximum retries reached for {keyword}. Skipping...")
                        break

                    print(f"Retrying in {wait_time} seconds...")
                    # time.sleep(wait_time)
                    # wait_time *= 2

        return all_trends

    def plot_trend_data(self, data, keywords_to_plot=None, title="Google Trends Data"):
        plt.figure(figsize=(12, 6))
        if keywords_to_plot is None:
            keywords_to_plot = data.columns

        for keyword in keywords_to_plot:
            if keyword in data.columns:
                plt.plot(data.index, data[keyword], label=keyword)

        plt.xlabel("Date")
        plt.ylabel("Search Interest")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()


save_dir = "test_data"
os.makedirs(save_dir, exist_ok=True)

proxies = {
    'http': 'http://5.79.73.131:13150',
    'https': 'http://5.79.73.131:13150'
}

google_trend = GoogelTrends(proxies=["http://5.79.73.131:13150"])

keywords = [
    "CS:GO", "Counter-Strike", "Global Offensive", "CS2", "Valve CS",
    "CS:GO Major", "Major Championship", "tournament", "IEM Katowice", "ESL One",
    "PGL Major", "BLAST Premier", "DreamHack", "Faceit Major", "ESEA league",
    "CS:GO Finals", "CS:GO Playoffs", "Valve Major", "CS:GO esports", "CS2 update",
    "CS:GO qualifiers", "CS:GO playoff match", "CS:GO grand finals", "CS:GO patch",
    "Operation Riptide", "Operation Broken Fang", "Operation Hydra", "Operation Wildfire",
    "CS:GO skins", "CS:GO trade", "CS:GO market", "CS:GO knives", "CS:GO case",
    "CS:GO trading platform", "CS:GO keys", "CS:GO AK-47 skins", "CS:GO AWP skins",
    "CS:GO knives market", "CS:GO sticker", "CS:GO graffiti", "CS:GO VAC ban",
    "CS:GO hacking", "CS:GO ban detection", "CS2 VAC system", "CS:GO pro players",
    "CS:GO best team", "CS:GO news", "CS:GO highlights", "CS:GO funny moments",
    "CS:GO maps", "CS:GO Mirage", "CS:GO Dust 2", "CS:GO Inferno"
]

trend_data = google_trend.get_trends(["Counter-Strike 2"])

if not trend_data.empty:
    file_path = os.path.join(save_dir, "csgo_trend_data.csv")
    trend_data.to_csv(file_path)
    print(f"Trend data saved to: {file_path}")

    keywords_to_plot = ["CS:GO", "Counter-Strike", "CS:GO Major"]
    google_trend.plot_trend_data(
        trend_data, keywords_to_plot, title="Search Interest Over Time"
    )
