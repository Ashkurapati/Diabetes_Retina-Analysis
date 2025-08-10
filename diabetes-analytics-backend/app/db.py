import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv, find_dotenv

# Load .env (works locally)
env_path = find_dotenv()
if env_path:
    load_dotenv(env_path)
else:
    load_dotenv()

MYSQL_URI = os.getenv("MYSQL_URI")
if not MYSQL_URI:
    raise RuntimeError(
        "MYSQL_URI is missing. Put it in a .env file or set it in the environment. "
        "Example: mysql+pymysql://root:Thulasi132003@127.0.0.1:3306/retina_data?charset=utf8mb4"
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
