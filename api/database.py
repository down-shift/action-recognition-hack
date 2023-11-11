from typing import Any

from sqlalchemy import create_engine, MetaData, JSON
from sqlalchemy.orm import declarative_base, DeclarativeBase

engine = create_engine("sqlite:///data.db", echo=True)


class Base(DeclarativeBase):
    type_annotation_map = {
        dict[str, Any]: JSON
    }
