from datetime import datetime
from typing import Any

from sqlalchemy import JSON
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy_serializer import SerializerMixin

from database import Base


class Record(Base, SerializerMixin):
    __tablename__ = 'records'
    serialize_rules = ('-request_headers', )

    id: Mapped[int] = mapped_column(primary_key=True, index=True, autoincrement=True)
    uuid: Mapped[str] = mapped_column(unique=True, index=True)
    location: Mapped[str]
    date: Mapped[datetime]
    classes: Mapped[dict[str, Any]] = mapped_column(nullable=True)
    file_path: Mapped[str]
    request_headers: Mapped[dict[str, Any]]
    processing_time: Mapped[int] = mapped_column(nullable=True)
    video_len: Mapped[int]
    date_stamp: Mapped[datetime] = mapped_column(default=datetime.now)
