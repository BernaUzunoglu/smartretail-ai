import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()  # .env dosyasını yükle

class Config:
    # PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "/app/"))
    # Ortam değişkeninden al, yoksa bu dosyanın 2 üst klasörünü baz al
    PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2]))

    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = os.getenv("DB_PORT", "5432")  # Varsayılan PostgreSQL portu 5432
    DB_NAME = os.getenv("DB_NAME")

    # SQLAlchemy için bağlantı URL'si
    DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"