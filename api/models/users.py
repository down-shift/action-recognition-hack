# import datetime
#
# from sqlalchemy import Column, Integer, String, Boolean, JSON, ForeignKey, DateTime
# from sqlalchemy.orm import relationship, Mapped, mapped_column
# from sqlalchemy_serializer import SerializerMixin
#
# from main import Base
#
#
# class Role(Base, SerializerMixin):
#     __tablename__ = 'roles'
#
#     id: Mapped[int] = mapped_column(primary_key=True, index=True, autoincrement=True)
#     uuid: Mapped[str] = mapped_column(unique=True, index=True)
#     name: Mapped[str]
#     permissions: Mapped[JSON]
#
#
# class User(Base, SerializerMixin):
#     __tablename__ = "users"
#     serialize_rules = ('-password',)
#
#     id: Mapped[int] = mapped_column(primary_key=True, index=True, autoincrement=True)
#     uuid: Mapped[str] = mapped_column(unique=True, index=True)
#     login: Mapped[str]
#     password: Mapped[str]  # hashed password
#     is_active: Mapped[str] = mapped_column(default=True)
#     first_name: Mapped[str]
#     last_name: Mapped[str]
#     role_id: Mapped[int] = mapped_column(ForeignKey('Role.id'))
#     registered_at: Mapped[datetime] = mapped_column(default=datetime.datetime.now)
