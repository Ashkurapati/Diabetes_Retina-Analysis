import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Expect the full MySQL URL in an env var (works locally via .env and on Render)
MYSQL_URI = os.getenv("MYSQL_URI")
if not MYSQL_URI:
    raise RuntimeError(
        "MYSQL_URI environment variable is missing. Set it to your hosted database URL "
        "(e.g., mysql+pymysql://USER:PASSWORD@HOST:3306/DB?charset=utf8mb4)."
    )

engine = create_engine(MYSQL_URI, pool_pre_ping=True, pool_recycle=3600)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
