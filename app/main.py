# app/main.py
import json
import io
import logging
from typing import List, Optional
from dotenv import load_dotenv
import os

from fastapi import (
    FastAPI,
    Request,
    UploadFile,
    File,
    Form,
    Depends,
    HTTPException,
    Response,
)
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from PyPDF2 import PdfReader

from .db import Base, engine, SessionLocal
from . import models, schemas, auth
from .chunker import split_into_chunks
from .embeddings import embed_texts
from .retrieval import search_chunks
from .llm import ask_gpt

load_dotenv()

# --- setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ida.backend")

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Intelligent Documentation Assistant (IDA) API")

# Allow credentials so browser can send/receive cookies.
# You should set allow_origins to your frontend origin in production.
FRONTEND_ORIGINS = os.getenv("IDA_FRONTEND_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Cookie settings
COOKIE_NAME = os.getenv("IDA_COOKIE_NAME", "ida_token")
REFRESH_COOKIE_NAME = os.getenv("IDA_REFRESH_NAME", "ida_refresh")
# True in prod (HTTPS). For local dev on http://localhost set to False
COOKIE_SECURE = os.getenv("IDA_COOKIE_SECURE", "0") == "1"
COOKIE_SAMESITE = os.getenv("IDA_COOKIE_SAMESITE", "lax")  # 'lax' is sensible default
ACCESS_EXPIRE_SECONDS = int(os.getenv("IDA_ACCESS_MIN", 15)) * 60
REFRESH_EXPIRE_SECONDS = int(os.getenv("IDA_REFRESH_DAYS", 7)) * 24 * 3600


# -------------------------
# DB dependency
# -------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/")
def read_root():
    return {"ok": True, "message": "Backend running. Visit /docs for API docs."}


# -------------------------
# Helpers to extract token
# -------------------------
def _extract_token_from_request(request: Request) -> Optional[str]:
    # Prefer Authorization header
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.lower().startswith("bearer "):
        return auth_header.split(" ", 1)[1]
    # Fallback to cookie
    cookie_token = request.cookies.get(COOKIE_NAME)
    if cookie_token:
        return cookie_token
    return None


# -------------------------
# Auth guards (accept header OR cookie)
# -------------------------
def require_org_admin(request: Request, db: Session = Depends(get_db)):
    token = _extract_token_from_request(request)
    if not token:
        raise HTTPException(status_code=401, detail="Missing auth token")
    payload = auth.decode_token(token)
    if not payload or payload.get("role") != "org_admin":
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    user_id = payload.get("user_id")
    org_id = payload.get("org_id")
    user = db.query(models.OrgUser).filter(
        models.OrgUser.id == user_id, models.OrgUser.org_id == org_id
    ).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token user")
    return {"org_id": org_id, "user_id": user_id}


def require_employee(request: Request, db: Session = Depends(get_db)):
    token = _extract_token_from_request(request)
    if not token:
        raise HTTPException(status_code=401, detail="Missing auth token")
    payload = auth.decode_token(token)
    if not payload or payload.get("role") != "employee":
        raise HTTPException(status_code=401, detail="Invalid token")
    emp_id = payload.get("employee_id")
    org_id = payload.get("org_id")
    emp = db.query(models.Employee).filter(
        models.Employee.id == emp_id, models.Employee.org_id == org_id
    ).first()
    if not emp or not emp.approved:
        raise HTTPException(status_code=403, detail="Not approved or invalid employee")
    return {"employee": emp, "org_id": org_id}


# -------------------------
# Organization endpoints
# -------------------------
@app.post("/org/signup")
def org_signup(body: schemas.OrgSignup, db: Session = Depends(get_db)):
    existing = db.query(models.Organization).filter(models.Organization.name == body.name).first()
    if existing:
        raise HTTPException(status_code=400, detail="Organization already exists")

    pw = str(body.password)[:72]
    org = models.Organization(name=body.name)
    db.add(org)
    db.flush()

    pw_hash = auth.hash_password(pw)
    admin = models.OrgUser(org_id=org.id, email=body.email, password_hash=pw_hash, is_admin=True)
    db.add(admin)
    db.commit()

    # Create tokens and return org info (but for browser sessions we prefer cookies)
    access = auth.create_access_token({"role": "org_admin", "org_id": org.id, "user_id": admin.id})
    refresh = auth.create_refresh_token({"role": "org_admin", "org_id": org.id, "user_id": admin.id})
    return {"ok": True, "token": access, "refresh": refresh, "org": {"id": org.id, "name": org.name}}


@app.post("/org/login")
def org_login(body: schemas.OrgLogin, response: Response, request: Request, db: Session = Depends(get_db)):
    user = db.query(models.OrgUser).filter(models.OrgUser.email == body.email).first()
    if not user or not auth.verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    org = db.query(models.Organization).filter(models.Organization.id == user.org_id).first()

    access = auth.create_access_token({"role": "org_admin", "org_id": org.id, "user_id": user.id})
    refresh = auth.create_refresh_token({"role": "org_admin", "org_id": org.id, "user_id": user.id})

    # Set cookies (httpOnly so JS cannot read them)
    response.set_cookie(
        COOKIE_NAME,
        access,
        max_age=ACCESS_EXPIRE_SECONDS,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite=COOKIE_SAMESITE,
        path="/",
    )
    response.set_cookie(
        REFRESH_COOKIE_NAME,
        refresh,
        max_age=REFRESH_EXPIRE_SECONDS,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite=COOKIE_SAMESITE,
        path="/auth/refresh",
    )

    # Return org info so frontend can show org name (frontend should NOT store token)
        # Return org info + token so JS (dev) can persist a token if needed
    return {
        "ok": True,
        "token": access,
        "refresh": refresh,
        "org": {"id": org.id, "name": org.name},
    }



@app.post("/org/logout")
def org_logout(response: Response):
    # Clear cookies in browser
    response.delete_cookie(COOKIE_NAME, path="/")
    response.delete_cookie(REFRESH_COOKIE_NAME, path="/auth/refresh")
    return {"ok": True}


# -------------------------
# Employee endpoints
# -------------------------
@app.post("/employee/signup")
def employee_signup(body: schemas.EmployeeSignup, db: Session = Depends(get_db)):
    org = db.query(models.Organization).filter(models.Organization.id == body.org_id).first()
    if not org:
        raise HTTPException(status_code=400, detail="Organization not found")

    # prevent duplicates
    existing = db.query(models.Employee).filter(
        models.Employee.org_id == body.org_id,
        (models.Employee.email == body.email) | (models.Employee.employee_id == body.employee_id),
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="Employee already registered for this org")

    emp = models.Employee(
        org_id=body.org_id,
        name=body.name,
        email=body.email,
        employee_id=body.employee_id,
        approved=False,
    )
    db.add(emp)
    db.commit()
    db.refresh(emp)
    return {"ok": True, "employee_id": emp.id}


@app.post("/employee/login")
def employee_login(body: schemas.EmployeeLogin, response: Response, request: Request, db: Session = Depends(get_db)):
    emp = db.query(models.Employee).filter(
        models.Employee.employee_id == body.employee_id,
        models.Employee.org_id == body.org_id,
    ).first()
    if not emp:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    access = auth.create_access_token({"role": "employee", "org_id": emp.org_id, "employee_id": emp.id})
    refresh = auth.create_refresh_token({"role": "employee", "org_id": emp.org_id, "employee_id": emp.id})

    response.set_cookie(
        COOKIE_NAME,
        access,
        max_age=ACCESS_EXPIRE_SECONDS,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite=COOKIE_SAMESITE,
        path="/",
    )
    response.set_cookie(
        REFRESH_COOKIE_NAME,
        refresh,
        max_age=REFRESH_EXPIRE_SECONDS,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite=COOKIE_SAMESITE,
        path="/auth/refresh",
    )

    return {
        "ok": True,
        "token": access,
        "refresh": refresh,
        "approved": bool(emp.approved),
    }



# -------------------------
# Refresh / Logout endpoints
# -------------------------
@app.post("/auth/refresh")
def auth_refresh(response: Response, request: Request, db: Session = Depends(get_db)):
    # Refresh token should be in cookie (or Authorization header)
    token = request.cookies.get(REFRESH_COOKIE_NAME) or (
        lambda: (lambda h: h.split(" ", 1)[1] if h and h.lower().startswith("bearer ") else None)(request.headers.get("Authorization"))
    )()
    if not token:
        raise HTTPException(status_code=401, detail="Missing refresh token")

    payload = auth.decode_token(token)
    if not payload or payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    # Build new access and refresh tokens (rotation)
    # Keep same claims but update token expiry
    new_access = auth.create_access_token({k: v for k, v in payload.items() if k not in ("exp", "type")})
    new_refresh = auth.create_refresh_token({k: v for k, v in payload.items() if k not in ("exp", "type")})

    response.set_cookie(
        COOKIE_NAME,
        new_access,
        max_age=ACCESS_EXPIRE_SECONDS,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite=COOKIE_SAMESITE,
        path="/",
    )
    response.set_cookie(
        REFRESH_COOKIE_NAME,
        new_refresh,
        max_age=REFRESH_EXPIRE_SECONDS,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite=COOKIE_SAMESITE,
        path="/auth/refresh",
    )

    return {"ok": True}


# -------------------------
# Org admin actions
# -------------------------
@app.get("/org/employees")
def org_employees(info=Depends(require_org_admin), db: Session = Depends(get_db)):
    org_id = info["org_id"]
    employees = (
        db.query(models.Employee)
        .filter(models.Employee.org_id == org_id)
        .order_by(models.Employee.created_at.desc())
        .all()
    )
    return {
        "ok": True,
        "employees": [
            {
                "id": e.id,
                "name": e.name,
                "email": e.email,
                "employee_id": e.employee_id,
                "approved": e.approved,
            }
            for e in employees
        ],
    }


@app.post("/org/employees/{emp_id}/approve")
def approve_emp(emp_id: int, info=Depends(require_org_admin), db: Session = Depends(get_db)):
    emp = (
        db.query(models.Employee)
        .filter(models.Employee.id == emp_id, models.Employee.org_id == info["org_id"])
        .first()
    )
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")
    emp.approved = True
    db.commit()
    return {"ok": True, "message": f"Employee {emp.name} approved."}


# -------------------------
# Document management
# -------------------------
@app.get("/org/docs")
def org_docs(info=Depends(require_org_admin), db: Session = Depends(get_db)):
    org_id = info["org_id"]
    docs = (
        db.query(models.Document)
        .filter(models.Document.org_id == org_id)
        .order_by(models.Document.created_at.desc())
        .all()
    )
    return {"ok": True, "docs": [{"id": d.id, "title": d.title, "source": d.source} for d in docs]}


@app.delete("/org/docs/{doc_id}")
def org_delete_doc(doc_id: int, info=Depends(require_org_admin), db: Session = Depends(get_db)):
    doc = (
        db.query(models.Document)
        .filter(models.Document.id == doc_id, models.Document.org_id == info["org_id"])
        .first()
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    db.delete(doc)
    db.commit()
    return {"ok": True, "message": "Document deleted."}


# -------------------------
# Ingest / Query (keeps your page number ingestion)
# -------------------------
@app.post("/ingest")
async def ingest(
    files: List[UploadFile] = File(...),
    request: Request = None,
    db: Session = Depends(get_db)
):
    org_id = None
    if request:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.lower().startswith("bearer "):
            payload = auth.decode_token(auth_header.split(" ", 1)[1])
            if payload and payload.get("role") == "org_admin":
                org_id = payload.get("org_id")
        # also accept cookie
        cookie_token = request.cookies.get(COOKIE_NAME)
        if not org_id and cookie_token:
            payload = auth.decode_token(cookie_token)
            if payload and payload.get("role") == "org_admin":
                org_id = payload.get("org_id")

    created = 0

    for f in files:
        data = await f.read()
        try:
            if f.filename.lower().endswith(".pdf"):
                reader = PdfReader(io.BytesIO(data))

                # Create one Document entry per PDF
                doc = models.Document(
                    org_id=org_id,
                    title=f.filename,
                    source=f.filename
                )
                db.add(doc)
                db.flush()

                # Loop through pages and create chunks with page numbers
                for pageno, page in enumerate(reader.pages, start=1):
                    page_text = page.extract_text() or ""
                    if not page_text.strip():
                        continue
                    chunks = split_into_chunks(page_text)
                    if chunks:
                        vecs = embed_texts(chunks)
                        for i, (ct, v) in enumerate(zip(chunks, vecs), start=1):
                            db.add(
                                models.Chunk(
                                    document_id=doc.id,
                                    text=ct,
                                    embedding_json=json.dumps(v),
                                    order=i,
                                    page=pageno,
                                )
                            )
                created += 1

            else:
                # Handle plain text or other files
                try:
                    text = data.decode("utf-8")
                except Exception:
                    text = data.decode("latin-1", errors="ignore")

                doc = models.Document(
                    org_id=org_id,
                    title=f.filename,
                    source=f.filename
                )
                db.add(doc)
                db.flush()

                chunks = split_into_chunks(text)
                if chunks:
                    vecs = embed_texts(chunks)
                    for i, (ct, v) in enumerate(zip(chunks, vecs), start=1):
                        db.add(
                            models.Chunk(
                                document_id=doc.id,
                                text=ct,
                                embedding_json=json.dumps(v),
                                order=i,
                                page=None,
                            )
                        )
                created += 1

        except Exception as e:
            logger.warning(f"⚠️ Failed to ingest {f.filename}: {e}")
            continue

    db.commit()
    return {"ok": True, "documents_added": created}


@app.post("/query")
async def query(
    request: Request,
    q: str = Form(...),
    k: int = Form(8),
    db: Session = Depends(get_db)
):
    """
    Query only the documents belonging to the organization
    whose token is provided in the Authorization header or cookie.
    """
    try:
        # --- extract org_id from token ---
        org_id = None
        token = _extract_token_from_request(request)
        if token:
            payload = auth.decode_token(token)
            if payload and payload.get("role") in ("org_admin", "employee"):
                org_id = payload.get("org_id")

        if not org_id:
            raise HTTPException(status_code=401, detail="Missing or invalid token")

        # --- Perform search scoped to org_id ---
        rows = search_chunks(db, q, k=k, org_id=org_id)
        if not rows:
            return {
                "ok": True,
                "answer": "No documents found for your organization. Please upload first.",
                "sources": [],
            }

        # --- Deduplicate by (title, source, page) ---
        seen = set()
        unique_chunks = []
        for r in rows:
            key = (r.get("title"), r.get("source"), r.get("page"))
            if key in seen:
                continue
            seen.add(key)
            unique_chunks.append(r)

        chosen = unique_chunks[:k]

        context = "\n\n".join([
            f"[{c['title']} | page: {c.get('page') or 'N/A'} | score:{round(c.get('score',0),3)}]\n{c['text']}"
            for c in chosen
        ])

        prompt = (
            "You are an Intelligent Documentation Assistant. "
            "Answer based **only** on the provided context from the organization's documents. "
            "If the context doesn't contain relevant information, say so.\n\n"
            f"Context:\n{context}\n\nQuestion: {q}"
        )

        answer = ask_gpt(prompt)

        sources = [
            {
                "title": c["title"],
                "source": c["source"],
                "page": c.get("page"),
                "score": c.get("score"),
            }
            for c in chosen
        ]

        return {"ok": True, "answer": answer, "sources": sources}

    except Exception as e:
        logger.exception("Query failed: %s", e)
        raise HTTPException(status_code=500, detail="Query failed.")
