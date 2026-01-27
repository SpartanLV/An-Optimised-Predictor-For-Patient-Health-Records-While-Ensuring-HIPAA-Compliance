import os
import time
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
if not DATABASE_URL:
    # Local fallback (dev)
    DATABASE_URL = "sqlite+pysqlite:///./data/patient_store.db"

connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, echo=False, future=True, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class Base(DeclarativeBase):
    pass


def init_db_with_retry(max_tries: int = 30, sleep_s: float = 1.0) -> None:
    """
    In docker compose, Postgres may not be ready when the API starts.
    This does a light retry by attempting a connection and then creating tables.
    """
    from .models import Patient  # noqa: F401 (ensure models imported)

    last_err = None
    for _ in range(max_tries):
        try:
            with engine.connect() as conn:
                conn.exec_driver_sql("SELECT 1")
            Base.metadata.create_all(bind=engine)
            return
        except Exception as e:
            last_err = e
            time.sleep(sleep_s)
    if last_err:
        raise last_err
