from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
from .db import Base

class Organization(Base):
    __tablename__ = "organizations"
    id = Column(Integer, primary_key=True)
    name = Column(String(256), unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    users = relationship("OrgUser", backref="organization", cascade="all, delete-orphan")
    employees = relationship("Employee", backref="organization", cascade="all, delete-orphan")
    documents = relationship("Document", backref="organization", cascade="all, delete-orphan")


class OrgUser(Base):
    __tablename__ = "org_users"
    id = Column(Integer, primary_key=True)
    org_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True)
    email = Column(String(256), unique=True, index=True)
    password_hash = Column(String(512))
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class Employee(Base):
    __tablename__ = "employees"
    id = Column(Integer, primary_key=True)
    org_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"))
    name = Column(String(256))
    email = Column(String(256))
    employee_id = Column(String(128), index=True)
    approved = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True)
    org_id = Column(Integer, ForeignKey("organizations.id", ondelete="SET NULL"), nullable=True)
    title = Column(String(1024))
    source = Column(String(1024))
    created_at = Column(DateTime, default=datetime.utcnow)

class Chunk(Base):
    __tablename__ = "chunks"
    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"))
    text = Column(Text)
    embedding_json = Column(Text)
    order = Column(Integer)
    page = Column(Integer, nullable=True)
