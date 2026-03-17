import logging
import os

from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.pool import NullPool
from sqlalchemy.sql import func

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL and "sqlite" in DATABASE_URL:
    engine = create_engine(
        DATABASE_URL, poolclass=NullPool, connect_args={"check_same_thread": False}
    )
else:
    engine = create_engine(DATABASE_URL, pool_size=20, max_overflow=50, pool_timeout=10)

db_session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))
Base = declarative_base()
Base.query = db_session.query_property()


class AutoExitTrade(Base):
    """
    Live auto-exit tracker for real trading.
    Created after an entry order is placed; updated when filled; closed when SL/TP hit.
    """

    __tablename__ = "auto_exit_trades"

    id = Column(Integer, primary_key=True)
    user_id = Column(String(255), nullable=False)          # OpenAlgo username
    strategy_name = Column(String(255), nullable=False)    # Strategy label sent in order payload
    exchange = Column(String(20), nullable=False)
    symbol = Column(String(80), nullable=False)
    product = Column(String(10), nullable=False)           # MIS/CNC/NRML
    action = Column(String(10), nullable=False)            # BUY/SELL
    quantity = Column(Integer, nullable=False)

    entry_orderid = Column(String(64), nullable=False)
    entry_price = Column(Float, nullable=True)
    stop_loss_price = Column(Float, nullable=True)
    take_profit_price = Column(Float, nullable=True)

    status = Column(String(30), nullable=False, default="await_fill")  # await_fill|active|closed|error
    exit_orderid = Column(String(64), nullable=True)
    error_message = Column(String(500), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


def init_db():
    from database.db_init_helper import init_db_with_logging

    init_db_with_logging(Base, engine, "Auto Exit DB", logger)

