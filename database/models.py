from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Boolean,
    DateTime,
    Text,
    ForeignKey,
    SmallInteger,
    DECIMAL,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy_utils import drop_database, create_database, database_exists
from datetime import datetime
import os
from dotenv import load_dotenv


load_dotenv()

Base = declarative_base()


def get_database_url():
    return f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"

def create_engine_with_config():
    return create_engine(get_database_url())

def create_session():
    engine = create_engine_with_config()
    Session = sessionmaker(bind=engine)
    return Session()

def reset_database():
    url = get_database_url()

    simple_url = url.replace("+psycopg2", "")

    if database_exists(simple_url):
        drop_database(simple_url)

    create_database(simple_url)

    engine = create_engine_with_config()
    Base.metadata.create_all(engine)
    

class Category(Base):
    __tablename__ = "categories"

    category_id = Column(SmallInteger, primary_key=True, autoincrement=True)
    category_name = Column(String(100), unique=True, nullable=False)

    items = relationship("Item", back_populates="category")


class Weapon(Base):
    __tablename__ = "weapons"

    weapon_id = Column(SmallInteger, primary_key=True, autoincrement=True)
    weapon_name = Column(String(100), unique=True, nullable=False)

    items = relationship("Item", back_populates="weapon")


class Collection(Base):
    __tablename__ = "collections"

    collection_id = Column(SmallInteger, primary_key=True, autoincrement=True)
    collection_name = Column(String(100), unique=True, nullable=False)

    items = relationship("Item", back_populates="collection")


class Rarity(Base):
    __tablename__ = "rarities"

    rarity_id = Column(SmallInteger, primary_key=True, autoincrement=True)
    rarity_name = Column(String(50), unique=True, nullable=False)

    items = relationship("Item", back_populates="rarity")


class Finish(Base):
    __tablename__ = "finishes"

    finish_id = Column(SmallInteger, primary_key=True, autoincrement=True)
    finish_name = Column(String(100), unique=True, nullable=False)

    items = relationship("Item", back_populates="finish")


class FinishStyle(Base):
    __tablename__ = "finish_styles"

    finish_style_id = Column(SmallInteger, primary_key=True, autoincrement=True)
    finish_style_name = Column(String(50), unique=True, nullable=False)

    items = relationship("Item", back_populates="finish_style")


class Team(Base):
    __tablename__ = "teams"

    team_id = Column(SmallInteger, primary_key=True, autoincrement=True)
    team_name = Column(String(100), unique=True, nullable=False)

    items = relationship("Item", back_populates="team")


class Tournament(Base):
    __tablename__ = "tournaments"

    tournament_id = Column(SmallInteger, primary_key=True, autoincrement=True)
    tournament_name = Column(String(100), unique=True, nullable=False)

    items = relationship("Item", back_populates="tournament")


class Pattern(Base):
    __tablename__ = "patterns"

    pattern_id = Column(SmallInteger, primary_key=True, autoincrement=True)
    pattern_name = Column(String(100), nullable=False)

    wear_ranges = relationship("WearRange", back_populates="pattern")


class WearRange(Base):
    __tablename__ = "wear_ranges"

    wear_range_id = Column(SmallInteger, primary_key=True, autoincrement=True)
    pattern_id = Column(SmallInteger, ForeignKey("patterns.pattern_id"), nullable=False)
    min_float_range = Column(DECIMAL(5, 4), nullable=False)
    max_float_range = Column(DECIMAL(5, 4), nullable=False)

    pattern = relationship("Pattern", back_populates="wear_ranges")
    items = relationship("ItemWearRange", back_populates="wear_range")


class Item(Base):
    __tablename__ = "items"

    item_id = Column(Integer, primary_key=True, autoincrement=True)
    hash_name = Column(String(255), nullable=False)
    pricempire_id = Column(Integer)
    pricempire_url = Column(Text)
    steam_url = Column(Text)
    item_image_url = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Foreign keys
    category_id = Column(SmallInteger, ForeignKey("categories.category_id"))
    weapon_id = Column(SmallInteger, ForeignKey("weapons.weapon_id"))
    collection_id = Column(SmallInteger, ForeignKey("collections.collection_id"))
    rarity_id = Column(SmallInteger, ForeignKey("rarities.rarity_id"))
    finish_id = Column(SmallInteger, ForeignKey("finishes.finish_id"))
    finish_style_id = Column(SmallInteger, ForeignKey("finish_styles.finish_style_id"))
    team_id = Column(SmallInteger, ForeignKey("teams.team_id"))
    tournament_id = Column(SmallInteger, ForeignKey("tournaments.tournament_id"))

    # Additional fields
    sticker_slots = Column(SmallInteger)
    doppler_phases = Column(SmallInteger)
    is_stattrak_available = Column(Boolean)
    is_souvenir_available = Column(Boolean)
    finish_style_catalog = Column(Integer)

    # Relationships
    category = relationship("Category", back_populates="items")
    weapon = relationship("Weapon", back_populates="items")
    collection = relationship("Collection", back_populates="items")
    rarity = relationship("Rarity", back_populates="items")
    finish = relationship("Finish", back_populates="items")
    finish_style = relationship("FinishStyle", back_populates="items")
    team = relationship("Team", back_populates="items")
    tournament = relationship("Tournament", back_populates="items")
    wear_ranges = relationship("ItemWearRange", back_populates="item")


class ItemWearRange(Base):
    __tablename__ = "item_wear_ranges"

    item_id = Column(Integer, ForeignKey("items.item_id"), primary_key=True)
    wear_range_id = Column(
        SmallInteger, ForeignKey("wear_ranges.wear_range_id"), primary_key=True
    )

    item = relationship("Item", back_populates="wear_ranges")
    wear_range = relationship("WearRange", back_populates="items")


class PriceTimeSeries(Base):
    __tablename__ = "price_time_series"

    # Primary key составной: item + provider + timestamp
    item_id = Column(Integer, ForeignKey("items.item_id"), primary_key=True)
    provider = Column(Text, primary_key=True)
    timestamp = Column(DateTime(timezone=True), primary_key=True)

    # Данные временного ряда
    price = Column(DECIMAL(8, 2), nullable=False)
    listings = Column(Integer, nullable=False)

    def __repr__(self):
        return (
            f"<PriceTimeSeries(item_id={self.item_id}, provider='{self.provider}', "
            f"timestamp={self.timestamp}, price={self.price}, listings={self.listings})>"
        )


# Функция для создания всех таблиц
def create_tables(engine=None):
    """Создает все таблицы в базе данных"""
    if engine is None:
        engine = create_engine_with_config()
    Base.metadata.create_all(engine)
    print("Таблицы успешно созданы!")


if __name__ == "__main__":
    # Подключение к базе данных используя настройки из .env
    engine = create_engine_with_config()
    
    # Создание таблиц
    create_tables(engine)
