# from sqlalchemy import create_engine
# from sqlalchemy.orm import sessionmaker
# from sqlalchemy.orm import declarative_base
# from core.config import Config
#
# # PostgreSQL bağlantısı için motoru oluştur
# engine = create_engine(Config.DATABASE_URL, echo=True)
#
# # Oturum yöneticisi oluştur
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
#
#
# # Declarative base sınıfı
# Base = declarative_base()
#
# # Dependency olarak kullanılacak fonksiyon - Her istek için yeni bir oturum açar ve işlem bitince kapatır.
# def get_db():
#     db = SessionLocal()core
#     try:
#         yield db
#     finally:
#         db.close()
#
# # Veritabanı bağlantısını test etme
# def test_connection():
#     try:
#         with engine.connect() as conn:
#             print("✅ PostgreSQL bağlantısı başarılı!")
#     except Exception as e:
#         print(f"❌ PostgreSQL bağlantı hatası: {e}")
#
# if __name__ == "__main__":
#     test_connection()