import time
import re
import json
import certifi
from selenium import webdriver
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from enum import Enum
from datetime import datetime
from fake_useragent import FakeUserAgent

from PricEmpireSaver import JSONSaver


class PriceHistoryTimePeriod(str, Enum):
    DAYS_7 = "7D"
    DAYS_30 = "30D"
    DAYS_90 = "90D"
    DAYS_180 = "180D"
    YEAR_1 = "1Y"
    ALL = "All"


class PriceHistoryProvider(str, Enum):
    BUFF163 = "buff163"
    CS_MONEY = "csmoney"
    SKINBARON = "skinbaron"
    TRADEIT_GG = "tradeitgg"
    BUFF_MARKET = "buffmarket"
    BITSKINS = "bitskins"
    WHITE_MARKET = "whitemarket"
    LOOTFARM = "lootfarm"
    CSFLOAT = "csfloat"
    SWAP_GG = "swapgg"
    STEAM = "steam"
    SKINBID = "skinbid"
    C5GAME = "c5game"
    YOUPIN898 = "youpin898"
    CS_TRADE = "cstrade"
    GAMERPAY = "gamerpay"
    LIS_SKINS = "lisskins"
    SHADOWPAY = "shadowpay"
    DMARKET = "dmarket"
    SKINSMONKEY = "skinsmonkey"
    SKINPORT = "skinport"
    CS_DEALS = "csdeals"
    SKINSWAP = "skinswap"
    MANNCO_STORE = "manncostore"
    CS_MONEY_MARKET = "csmoneymarket"
    SKINFLOW = "skinflow"
    SKINOUT = "skinout"
    HALOSKINS = "haloskins"
    RAPIDSKINS = "rapidskins"
    KRAKATOA = "krakatoa"
    MARKET_CSGO = "marketcsgo"
    BUFF163_BUY_ORDER = "buff163buyorder"
    STEAM_BUY_ORDER = "steambuyorder"
    DMARKET_BUY_ORDER = "dmarketbuyorder"
    MARKET_CSGO_BUY_ORDER = "marketcsgobuyorder"
    ITRADE_GG = "itradegg"
    SKINDECK = "skindeck"
    AVAN_MARKET = "avanmarket"
    YOUPIN898_BUY_ORDER = "youpin898buyorder"
    CSGOEMPIRE = "csgoempire"
    CSGO500 = "csgo500"
    CSGOFAST = "csgofast"
    CLASH_GG = "clashgg"
    CSGOROLL = "csgoroll"
    SKINRAVE = "skinrave"
    ROLLBIT = "rollbit"
    CSGOEMPIRE_STORE = "csgoempirestore"
    CSGOROLL_STORE = "csgorollstore"
    CSGOGEM = "csgogem"
    SKINS_49 = "49skins"
    SNIPESKINS = "snipeskins"
    RUSTTM = "rusttm"
    SKINVAULT = "skinvault"
    SKIN_PLACE = "skinplace"
    WAXPEER_BUY_ORDER = "waxpeerbuyorder"
    EXESKINS = "exeskins"
    ECOSTEAM = "ecosteam"
    ECOSTEAM_BUY_ORDER = "ecosteambuyorder"
    GAMEBOOST = "gameboost"
    SKINSWAP_CN = "skinswapcn"


PROVIDER_DISPLAY_NAMES = {
    PriceHistoryProvider.BUFF163: "Buff.163",
    PriceHistoryProvider.CS_MONEY: "CS.MONEY",
    PriceHistoryProvider.SKINBARON: "SkinBaron",
    PriceHistoryProvider.TRADEIT_GG: "TradeIt.GG",
    PriceHistoryProvider.BUFF_MARKET: "Buff.Market",
    PriceHistoryProvider.BITSKINS: "BitSkins",
    PriceHistoryProvider.WHITE_MARKET: "White.Market",
    PriceHistoryProvider.LOOTFARM: "LootFarm",
    PriceHistoryProvider.CSFLOAT: "CSFloat",
    PriceHistoryProvider.SWAP_GG: "Swap.GG",
    PriceHistoryProvider.STEAM: "Steam",
    PriceHistoryProvider.SKINBID: "SkinBid",
    PriceHistoryProvider.C5GAME: "C5Game",
    PriceHistoryProvider.YOUPIN898: "YouPin898",
    PriceHistoryProvider.CS_TRADE: "CS.Trade",
    PriceHistoryProvider.GAMERPAY: "GamerPay",
    PriceHistoryProvider.LIS_SKINS: "Lis-skins",
    PriceHistoryProvider.SHADOWPAY: "ShadowPay",
    PriceHistoryProvider.DMARKET: "DMarket",
    PriceHistoryProvider.SKINSMONKEY: "SkinsMonkey",
    PriceHistoryProvider.SKINPORT: "Skinport",
    PriceHistoryProvider.CS_DEALS: "CS.Deals",
    PriceHistoryProvider.SKINSWAP: "SkinSwap",
    PriceHistoryProvider.MANNCO_STORE: "Mannco.Store",
    PriceHistoryProvider.CS_MONEY_MARKET: "CS.MONEY Market",
    PriceHistoryProvider.SKINFLOW: "SkinFlow",
    PriceHistoryProvider.SKINOUT: "SkinOut",
    PriceHistoryProvider.HALOSKINS: "HaloSkins",
    PriceHistoryProvider.RAPIDSKINS: "RapidSkins",
    PriceHistoryProvider.KRAKATOA: "Krakatoa",
    PriceHistoryProvider.MARKET_CSGO: "Market.CSGO",
    PriceHistoryProvider.BUFF163_BUY_ORDER: "Buff.163 Buy Order",
    PriceHistoryProvider.STEAM_BUY_ORDER: "Steam Buy Order",
    PriceHistoryProvider.DMARKET_BUY_ORDER: "DMarket Buy Order",
    PriceHistoryProvider.MARKET_CSGO_BUY_ORDER: "Market.CSGO Buy Order",
    PriceHistoryProvider.ITRADE_GG: "iTrade.GG",
    PriceHistoryProvider.SKINDECK: "SkinDeck",
    PriceHistoryProvider.AVAN_MARKET: "Avan.Market",
    PriceHistoryProvider.YOUPIN898_BUY_ORDER: "YouPin898 Buy Order",
    PriceHistoryProvider.CSGOEMPIRE: "CSGOEmpire",
    PriceHistoryProvider.CSGO500: "CSGO500",
    PriceHistoryProvider.CSGOFAST: "CSGOFast",
    PriceHistoryProvider.CLASH_GG: "Clash.GG",
    PriceHistoryProvider.CSGOROLL: "CSGORoll",
    PriceHistoryProvider.SKINRAVE: "SkinRave",
    PriceHistoryProvider.ROLLBIT: "RollBit",
    PriceHistoryProvider.CSGOEMPIRE_STORE: "CSGOEmpire Store",
    PriceHistoryProvider.CSGOROLL_STORE: "CSGORoll Store",
    PriceHistoryProvider.CSGOGEM: "CSGOGem",
    PriceHistoryProvider.SKINS_49: "49skins",
    PriceHistoryProvider.SNIPESKINS: "SnipeSkins",
    PriceHistoryProvider.RUSTTM: "RustTM",
    PriceHistoryProvider.SKINVAULT: "SkinVault",
    PriceHistoryProvider.SKIN_PLACE: "Skin.place",
    PriceHistoryProvider.WAXPEER_BUY_ORDER: "Waxpeer Buy Order",
    PriceHistoryProvider.EXESKINS: "ExeSkins",
    PriceHistoryProvider.ECOSTEAM: "Ecosteam",
    PriceHistoryProvider.ECOSTEAM_BUY_ORDER: "Ecosteam Buy Order",
    PriceHistoryProvider.GAMEBOOST: "GameBoost",
    PriceHistoryProvider.SKINSWAP_CN: "SkinSwap CN",
}


DELAY = 10


class PriceEmpireParser:
    def __init__(self, path_to_links_file, cookies_path=None):
        self.chrome_options = webdriver.ChromeOptions()
        self.chrome_options.add_argument("--ignore-certificate-errors")
        self.chrome_options.add_argument("--ignore-ssl-errors")
        self.chrome_options.add_argument(f"--ssl-certificates-path={certifi.where()}")
        # self.chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        # self.chrome_options.add_argument('--disable-cache')
        # self.chrome_options.add_argument('--headless')
        # self.chrome_options.add_argument('--disable-gpu')
        # self.chrome_options.add_experimental_option("prefs", {"profile.managed_default_content_settings.images": 2,
        #                                                           "profile.managed_default_content_settings.stylesheet": 2,})
        self.chrome_options.set_capability("goog:loggingPrefs", {"performance": "ALL"})
        # self.chrome_options.add_experimental_option(
        #     "excludeSwitches", ["enable-automation", "enable-logging"]
        # )
        self.browser = uc.Chrome(options=self.chrome_options)

        self.path_to_links_file = path_to_links_file
        self.urls = self.load_urls()

        self.saver = JSONSaver("data/items_data.json")

        if cookies_path is not None:
            self.update_cookies(cookies_path)

    def __enter__(self):
        self.browser.maximize_window()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.browser.quit()
    
    def enable_performance_logging(self):
        caps = DesiredCapabilities.CHROME
        caps["goog:loggingPrefs"] = {"performance": "ALL"}
        return caps

    def load_urls(self):
        with open(self.path_to_links_file, encoding="utf-8") as file:
            data = json.load(file)
        unique_data = []
        seen = set()
        for item in data:
            if item not in seen:
                seen.add(item)
                unique_data.append(item)
        print(f"Загружено {len(unique_data)}/{len(data)} url")
        return unique_data

    def update_cookies(self, cookies_path):
        with open(cookies_path, "r", encoding="utf-8") as file:
            cookies_string = file.read()
            self.browser.get("https://pricempire.com")
            self.browser.delete_all_cookies()
            cookies_list = cookies_string.split(";")
            for cookie in cookies_list:
                cookie = cookie.strip()
                if "=" in cookie:
                    name, value = cookie.split("=", 1)
                    cookie_dict = {
                        "name": name.strip(),
                        "value": value.strip(),
                        "domain": ".pricempire.com",
                        "path": "/",
                        "secure": True,
                        "httpOnly": False,
                    }

                    self.browser.add_cookie(cookie_dict)
            self.browser.refresh()
    
    def parse(self, time_period: PriceHistoryTimePeriod = PriceHistoryTimePeriod.ALL, 
            providers = [PriceHistoryProvider.BUFF163], batch_size=100):
        existing_items = self.saver.get_all_items()
        existing_urls = {item.get("url") for item in existing_items if item.get("url")}

        batch = []
        
        for i, url in enumerate(self.urls):
            if url in existing_urls:
                print(f"Пропускаем уже обработанный URL: {url}")
                continue
            try:
                print(f"Парсим URL ({i+1}/{len(self.urls)}): {url}")
                item_data = self.parse_page(url, time_period, providers)
                print(f"Parsing OK {url}")
                item_data["url"] = url
                batch.append(item_data)
                
                if len(batch) >= batch_size:
                        self.saver.save_items_batch(batch)
                        print(f"Сохранена пачка из {len(batch)} предметов")
                        batch = []
                
            except Exception as e:
                print(f"Ошибка при парсинге {url}: {e}")
                continue
        
        if batch:
            self.saver.save_items_batch(batch)
            print(f"Сохранена пачка из {len(batch)} предметов")

    def parse_page(
        self, url, time_period: PriceHistoryTimePeriod = PriceHistoryTimePeriod.ALL, providers = [PriceHistoryProvider.BUFF163]):
        self.browser.get(url)

        # error_elem = self.browser.find_elements(By.ID, "splash-error")
        # if error_elem:
        #     print("captcha FOUND!!!!")
        #     captcha_locator = (By.CSS_SELECTOR, 'input[type="checkbox"]')
        #     captcha = WebDriverWait(self.browser, DELAY).until(
        #         EC.element_to_be_clickable(captcha_locator)
        #     )
        #     print("Captcha element found")

        #     # Кликаем по чекбоксу через JS (обычный клик часто блокируется)
        #     self.browser.execute_script("arguments[0].click();", captcha)
        #     print("Checkbox clicked via JS")

        item_data = {}

        header_locator = (By.CSS_SELECTOR, "div.flex.items-center h1")
        WebDriverWait(self.browser, DELAY).until(EC.presence_of_element_located(header_locator))

        item_name = self.browser.find_element(*header_locator).text.strip()
        item_data["item_name"] = item_name

        general_stats_block = self.browser.find_element(By.CSS_SELECTOR, "div.hidden.items-center")
        total_offers, trades_7d, liquidity, rank = [el.text.strip() for el in general_stats_block.find_elements(By.CSS_SELECTOR, "p.flex")]
        item_data["offers"] = int(total_offers.rstrip("\noffers").replace(",", ""))
        item_data["trades/7d"] = int(trades_7d.removesuffix("\ntrades/7d").replace(",", ""))
        item_data["liquidity"] = float(liquidity.rstrip("\nliquidity%"))
        item_data["rank"] = int(rank.rstrip("\nrank").lstrip("#\n").replace(",", ""))

        steam_url_elem = self.browser.find_element(By.CSS_SELECTOR, "div.flex.items-center.justify-between.p-4 a[href]")
        steam_url = steam_url_elem.get_attribute("href")
        item_data["steam_url"] = steam_url

        item_image_url = self.browser.find_element(By.CSS_SELECTOR, "div.relative.flex.h-80.w-full.items-center img")
        item_data["item_image_url"] = item_image_url.get_attribute("src")

        # total_variants = self.browser.find_element(By.CSS_SELECTOR, ".flex.flex-col.gap-2.rounded-b-lg.backdrop-blur-sm span.rounded-md")
        # total_variants_num = int(total_variants.text.rstrip("variants"))
        # item_data["variants"] = total_variants_num

        variants = {}
        variants_locator = (By.ID, "variants-heading")
        WebDriverWait(self.browser, DELAY).until(EC.presence_of_element_located(variants_locator))
        variants_list = self.browser.find_element(By.CSS_SELECTOR, 'div.flex.flex-col[role="list"]')
        for variant in variants_list.find_elements(By.TAG_NAME, "a"):
            name_parts = variant.find_elements(By.CSS_SELECTOR, "p.flex.items-center.gap-2 span")
            name = " ".join(elem.text.strip() for elem in name_parts)

            price_element = variant.find_element(By.CSS_SELECTOR, "span.font-bold.text-theme-200")
            price_text = price_element.text.strip()
            price = float(price_text.replace('$', '').replace(',', ''))

            variants[name] = price
        
        item_data["Variants"] = variants

        data_wear_range = {}
        locator_wear_range = (By.ID, "pattern-name-label")
        # WebDriverWait(self.browser, DELAY).until(EC.presence_of_element_located(locator_wear_range))
        pattern_names = self.browser.find_elements(*locator_wear_range)
        if pattern_names:
            pattern_name = pattern_names[0]
            wear_locator = (By.CSS_SELECTOR, '[aria-labelledby="pattern-name-label"]')
            WebDriverWait(self.browser, DELAY).until(EC.presence_of_element_located(wear_locator))
            data_wear_range["Pattern Name"] = self.browser.find_element(*wear_locator).text.strip()

            float_range = self.browser.find_element(By.ID, "float-range-label").parent
            number_spans = float_range.find_elements(By.CSS_SELECTOR, "span.rounded-md.bg-theme-700.font-mono")
            float_min = number_spans[0].text.strip()
            float_max = number_spans[1].text.strip()
            float_range_str = f"{float_min} - {float_max}"
            data_wear_range["Float Range"] = float_range_str

        item_data["Wear Range"] = data_wear_range

        data_overview = {}
        locator_overview = (By.CSS_SELECTOR, 'section.order-1.flex.w-full.flex-col.gap-4[aria-label*="Detailed information"] .grid[aria-label*="Overview details"]')
        WebDriverWait(self.browser, DELAY).until(EC.presence_of_element_located(locator_overview))
        overview = self.browser.find_element(*locator_overview)
        for tag in overview.find_elements(By.TAG_NAME, "div"):
            if tag.get_attribute("style") != "display:none;":
                dd_element = tag.find_element(By.TAG_NAME, "dd")
                dd_text = dd_element.text.strip()
                if dd_text:
                    key = tag.find_element(By.TAG_NAME, "dt").text.strip()
                    data_overview[key] = dd_text
        item_data["Overview"] = data_overview

        price_statistics = {}
        locator_price_stat = (By.CSS_SELECTOR, '[aria-label*="Price statistics details"]')
        # WebDriverWait(self.browser, DELAY).until(EC.presence_of_element_located(locator_price_stat))
        price_stats = self.browser.find_elements(*locator_price_stat)
        if price_stats:
            price_stats = price_stats[0]
            for tag in price_stats.find_elements(By.TAG_NAME, "div"):
                if tag.get_attribute("style") != "display:none;":
                    dd_element = tag.find_element(By.TAG_NAME, "dd")
                    dd_text = dd_element.text.strip()
                    if dd_text:
                        key = tag.find_element(By.TAG_NAME, "dt").text.strip()
                        price_statistics[key] = float(dd_text.lstrip("$").replace(",", ""))
        item_data["Price Statistics"] = price_statistics

        trade_statistics = {}
        locator_trade_statistics = (By.CSS_SELECTOR, '[aria-describedby*="trade-stats-description"]')
        # WebDriverWait(self.browser, DELAY).until(EC.presence_of_element_located(locator_trade_statistics))
        trade_stats = self.browser.find_elements(*locator_trade_statistics)
        if trade_stats:
            trade_stats = trade_stats[0]
            for tag in trade_stats.find_elements(By.TAG_NAME, "div"):
                if tag.get_attribute("style") != "display:none;":
                    dd_element = tag.find_element(By.TAG_NAME, "dd")
                    dd_text = dd_element.text.strip()
                    if dd_text:
                        key = tag.find_element(By.TAG_NAME, "dt").text.strip()
                        trade_statistics[key] = dd_text
        item_data["Trade Statistics"] = trade_statistics
            
        locator = (By.CSS_SELECTOR, "div#multi-chart")
        WebDriverWait(self.browser, DELAY).until(EC.presence_of_element_located(locator))

        price_history_container = self.browser.find_element(
            By.CSS_SELECTOR, "div#multi-chart"
        )
        buttons = price_history_container.find_elements(
            By.CSS_SELECTOR,
            "div.space-y-6.p-4 div.relative.flex > button",
        )

        period_to_button = {
            PriceHistoryTimePeriod.DAYS_7: buttons[0],
            PriceHistoryTimePeriod.DAYS_30: buttons[1],
            PriceHistoryTimePeriod.DAYS_90: buttons[2],
            PriceHistoryTimePeriod.DAYS_180: buttons[3],
            PriceHistoryTimePeriod.YEAR_1: buttons[4],
            PriceHistoryTimePeriod.ALL: buttons[5],
        }

        target_button = period_to_button.get(time_period)
        self.browser.execute_script(
            "return arguments[0].scrollIntoView(true);", price_history_container
        )
        WebDriverWait(self.browser, DELAY).until(EC.element_to_be_clickable(target_button))
        self.browser.execute_script("arguments[0].click();", target_button)

        # select_providers_btn = price_history_container.find_element(By.CSS_SELECTOR, "div#multi-chart .flex.flex-grow")
        # self.browser.execute_script(
        #     "return arguments[0].scrollIntoView(true);", select_providers_btn
        # )
        # WebDriverWait(self.browser, DELAY).until(EC.element_to_be_clickable(select_providers_btn))
        # self.browser.execute_script("arguments[0].click();", select_providers_btn)

        # scroller_providers = self.browser.find_elements(By.CSS_SELECTOR, "div.max-h-60.overflow-y-auto.scrollbar-thin")[-1]
        # self.browser.execute_script("return arguments[0].scrollIntoView(true);", scroller_providers)
        # self.scroll_to_bottom(scroller_providers)

        # self.select_providers(scroller_providers, providers)
        pricempire_id = self.get_pricempire_item_id()
        item_data["pricempire_id"] = int(pricempire_id)

        print(item_data)

        return item_data

    def scroll_to_bottom(self, element):
            last_height = self.browser.execute_script("return arguments[0].scrollHeight", element)
            while True:
                self.browser.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", element)
                time.sleep(0.1)
                
                new_height = self.browser.execute_script("return arguments[0].scrollHeight", element)
                
                if new_height == last_height:
                    break
                last_height = new_height

    def select_providers(self, container, providers):
        desired_display_names = [PROVIDER_DISPLAY_NAMES[p] for p in providers]
        
        provider_elements = container.find_elements(By.CSS_SELECTOR, "div.select-option")
        
        for provider_element in provider_elements:
            provider_text = provider_element.text.strip()
            if provider_text in desired_display_names:
                self.browser.execute_script("arguments[0].click();", provider_element)
    
    def get_pricempire_item_id(self):
        self.browser.execute_cdp_cmd("Network.enable", {})
        logs = self.browser.get_log("performance")
        for entry in logs:
            msg = json.loads(entry["message"])["message"]
            if msg["method"] == "Network.requestWillBeSent":
                url = msg["params"]["request"]["url"]
                if url.startswith("https://pricempire.com/api-data"):
                    match = re.search(r'[?&]id=(\d+)', url)
                    return match.group(1)


if __name__ == "__main__":
    start = time.perf_counter()
    with PriceEmpireParser(
        path_to_links_file="pricempire_items_urls.json",
        cookies_path="pricempire_cookies.txt",
    ) as parser:
        parser.parse()
        # parser.parse_page("https://pricempire.com/cs2-items/skin/sawed-off-sage-spray/souvenir-field-tested")
    print(f"Время выполнения скрипта: {time.perf_counter() - start} секунд")
