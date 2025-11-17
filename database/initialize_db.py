from sqlalchemy import create_engine, text
from sqlalchemy_utils import database_exists, create_database, drop_database
from dotenv import load_dotenv
from models import Base, create_engine_with_config
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


load_dotenv()


def setup_database(reset_db=True):
    """Создание базы данных, обычных таблиц и hypertable для price_time_series"""
    engine = create_engine_with_config()
    simple_url = engine.url.set(drivername="postgresql")  # без psycopg2

    # Дропаем и создаем базу заново
    if reset_db and database_exists(simple_url):
        logger.info("Dropping existing database...")
        drop_database(simple_url)
    if not database_exists(simple_url):
        logger.info("Creating new database...")
        create_database(simple_url)

    # Создаем все обычные таблицы SQLAlchemy (items, wear_ranges, и т.д.)
    logger.info("Creating standard tables...")
    Base.metadata.create_all(engine)
    logger.info("✓ Standard tables created!")

    # Создание hypertable для временного ряда
    with engine.connect() as conn:
        # Проверка, есть ли TimescaleDB extension
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb;"))
        logger.info("✓ TimescaleDB extension enabled")
        
        # Создаем таблицу price_time_series, если она ещё не создана
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS price_time_series (
                item_id INTEGER NOT NULL,
                provider TEXT NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                price DECIMAL(8,2) NOT NULL,
                listings SMALLINT NOT NULL,
                PRIMARY KEY (item_id, provider, timestamp)
            );
        """))
        logger.info("✓ Price_time_series table created/verified")

        # Превращаем таблицу в hypertable (TimescaleDB)
        conn.execute(text("""
            SELECT create_hypertable('price_time_series', 'timestamp', if_not_exists => TRUE);
        """))

        # Дополнительно можно настроить chunk_time_interval (например, 1 день)
        conn.execute(text("""
            SELECT set_chunk_time_interval('price_time_series', INTERVAL '1 day');
        """))
        logger.info("✓ Hypertable created with 1-day chunks")

        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_price_series_item_timestamp ON price_time_series (item_id, timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_price_series_provider_timestamp ON price_time_series (provider, timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_price_series_timestamp ON price_time_series (timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_price_series_item_provider ON price_time_series (item_id, provider)"
        ]
        
        for idx_sql in indexes:
            conn.execute(text(idx_sql))
        logger.info("✓ Optimized indexes created")


if __name__ == "__main__":
    setup_database(reset_db=True)
