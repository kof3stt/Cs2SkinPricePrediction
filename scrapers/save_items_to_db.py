import json
import sys
import os
from sqlalchemy.orm import Session


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import (
    create_session,
    Category,
    Weapon,
    Collection,
    Rarity,
    Finish,
    FinishStyle,
    Team,
    Tournament,
    Pattern,
    WearRange,
    Item,
    ItemWearRange,
    reset_database,
)
from decimal import Decimal
from datetime import datetime


class BaseJsonParser:
    def __init__(self, json_path: str, reset_db=False):
        if reset_db:
            reset_database()
        self.json_path = json_path
        self.session: Session = create_session()
        self.items_data = self.load_json()

        self.cache = {
            Category: {},
            Weapon: {},
            Collection: {},
            Rarity: {},
            Finish: {},
            FinishStyle: {},
            Team: {},
            Tournament: {},
            Pattern: {},
            WearRange: {},
        }

    def load_json(self):
        with open(self.json_path, "r", encoding="utf-8") as f:
            return json.load(f)["items"]

    def get_or_create_relation(self, model, name_field: str, value: str):
        if not value:
            return None

        model_cache = self.cache.get(model, {})
        if value in model_cache:
            return model_cache[value]

        instance = (
            self.session.query(model)
            .filter(getattr(model, name_field) == value)
            .first()
        )

        if instance:
            model_cache[value] = instance
            return instance

        instance = model(**{name_field: value})
        self.session.add(instance)
        self.session.flush()
        model_cache[value] = instance
        return instance

    def parse_item(self, data: dict):
        overview = data.get("Overview", {})
        wear_data = data.get("Wear Range")

        category = self.get_or_create_relation(
            Category, "category_name", overview.get("Category")
        )
        weapon = self.get_or_create_relation(
            Weapon, "weapon_name", overview.get("Weapon")
        )
        collection = self.get_or_create_relation(
            Collection, "collection_name", overview.get("Collection")
        )
        rarity = self.get_or_create_relation(
            Rarity, "rarity_name", overview.get("Rarity")
        )
        finish = self.get_or_create_relation(
            Finish, "finish_name", overview.get("Finish")
        )
        finish_style = self.get_or_create_relation(
            FinishStyle, "finish_style_name", overview.get("Finish Style")
        )
        team = self.get_or_create_relation(Team, "team_name", overview.get("Team"))
        tournament = self.get_or_create_relation(
            Tournament, "tournament_name", overview.get("Tournament")
        )

        pricempire_id = data.get("pricempire_id")
        item = (
            self.session.query(Item).filter_by(pricempire_id=pricempire_id).first()
            if pricempire_id
            else None
        )

        stattrak_value = overview.get("StatTrak™ Available")
        souvenir_value = overview.get("Souvenir Available")
        finish_style_catalog_value = overview.get("Finish Style Catalog")

        if item:
            item.hash_name = data.get("item_name")
            item.pricempire_url = data.get("url")
            item.steam_url = data.get("steam_url")
            item.item_image_url = data.get("item_image_url")
            item.created_at = datetime.now()
            item.category = category
            item.weapon = weapon
            item.collection = collection
            item.rarity = rarity
            item.finish = finish
            item.finish_style = finish_style
            item.team = team
            item.tournament = tournament
            item.sticker_slots = (
                int(overview.get("Sticker Slots", 0))
                if overview.get("Sticker Slots")
                else None
            )
            item.doppler_phases = (
                int(overview.get("Doppler Phases", 0))
                if overview.get("Doppler Phases")
                else None
            )

            item.is_stattrak_available = (
                True
                if stattrak_value == "Yes"
                else False if stattrak_value == "No" else None
            )
            item.is_souvenir_available = (
                True
                if souvenir_value == "Yes"
                else False if souvenir_value == "No" else None
            )
            item.finish_style_catalog = (
                int(finish_style_catalog_value)
                if finish_style_catalog_value
                and str(finish_style_catalog_value).isdigit()
                else None
            )
        else:
            item = Item(
                hash_name=data.get("item_name"),
                pricempire_id=pricempire_id,
                pricempire_url=data.get("url"),
                steam_url=data.get("steam_url"),
                item_image_url=data.get("item_image_url"),
                created_at=datetime.now(),
                category=category,
                weapon=weapon,
                collection=collection,
                rarity=rarity,
                finish=finish,
                finish_style=finish_style,
                team=team,
                tournament=tournament,
                sticker_slots=(
                    int(overview.get("Sticker Slots", 0))
                    if overview.get("Sticker Slots")
                    else None
                ),
                doppler_phases=(
                    int(overview.get("Doppler Phases", 0))
                    if overview.get("Doppler Phases")
                    else None
                ),
                is_stattrak_available=(
                    True
                    if stattrak_value == "Yes"
                    else False if stattrak_value == "No" else None
                ),
                is_souvenir_available=(
                    True
                    if souvenir_value == "Yes"
                    else False if souvenir_value == "No" else None
                ),
                finish_style_catalog=(
                    int(finish_style_catalog_value)
                    if finish_style_catalog_value
                    and str(finish_style_catalog_value).isdigit()
                    else None
                ),
            )
            self.session.add(item)
            self.session.flush()

        if wear_data and isinstance(wear_data, dict):
            pattern_name = wear_data.get("Pattern Name")
            float_range_str = wear_data.get("Float Range")

            if pattern_name and float_range_str:
                pattern = self.get_or_create_relation(
                    Pattern, "pattern_name", pattern_name
                )

                min_str, max_str = float_range_str.split("-")
                min_float = Decimal(min_str.strip())
                max_float = Decimal(max_str.strip())

                if min_float is not None and max_float is not None:
                    wear_key = f"{pattern_name}:{min_float}-{max_float}"
                    wear_range = self.cache[WearRange].get(wear_key)

                    if not wear_range:
                        wear_range = (
                            self.session.query(WearRange)
                            .filter_by(
                                pattern_id=pattern.pattern_id,
                                min_float_range=min_float,
                                max_float_range=max_float,
                            )
                            .first()
                        )
                        if not wear_range:
                            wear_range = WearRange(
                                pattern=pattern,
                                min_float_range=min_float,
                                max_float_range=max_float,
                            )
                            self.session.add(wear_range)
                            self.session.flush()
                        self.cache[WearRange][wear_key] = wear_range

                    if not any(
                        iwr.wear_range_id == wear_range.wear_range_id
                        for iwr in item.wear_ranges
                    ):
                        item_wear_range = ItemWearRange(
                            item=item, wear_range=wear_range
                        )
                        self.session.add(item_wear_range)

    def parse_all(self):
        for data in self.items_data:
            self.parse_item(data)
        self.session.commit()
        print(f"{len(self.items_data)} items processed.")


if __name__ == "__main__":
    parser = BaseJsonParser("scrapers/data/items_data.json", reset_db=False)
    parser.parse_all()
