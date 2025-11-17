import subprocess
import os
from datetime import datetime
from dotenv import load_dotenv


load_dotenv()


def backup_database():
    """Создает резервную копию базы данных PostgreSQL/TimescaleDB."""
    db_name = os.getenv("DB_NAME", "cs")
    user = os.getenv("DB_USER", "cs_user")
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    password = os.getenv("DB_PASSWORD", "")
    output_dir = os.getenv("BACKUP_DIR", "backups")

    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/backup_{db_name}_{timestamp}.dump"

    env = os.environ.copy()
    env["PGPASSWORD"] = password

    command = [
        "pg_dump",
        "-U", user,
        "-h", host,
        "-p", port,
        "-d", db_name,
        "-F", "c",
        "-f", filename
    ]

    print(f"💾 Создаю резервную копию базы данных '{db_name}'...")
    subprocess.run(command, check=True, env=env)
    print(f"✅ Бэкап успешно создан: {filename}")


if __name__ == "__main__":
    backup_database()