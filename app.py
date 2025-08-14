"""
Freelancer CV Collector & Analyzer — Streamlit App (Admin + Candidate Form)
===========================================================================

New:
- Admin-only Dashboard (set ADMIN_PASSWORD in Secrets or env).
- Candidate intake fields: name, birthdate, email, nationality, oil&gas exp, notice period,
  work title, years of experience.
- Safe DB migration to add missing columns.

Local:
  pip install -r requirements.txt
  streamlit run app.py

Cloud:
- Add Secret: ADMIN_PASSWORD = "your-strong-pass"
- (Optional) CV_ENGINE_DB = "/mount/tmp/cv_engine.sqlite3"
"""

from __future__ import annotations
import re
import os
import io
import sqlite3
from datetime import datetime
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st

# Resume parsing libs
from pdfminer.high_level import extract_text as pdf_extract_text
import docx2txt

# NLP for similarity scoring
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# Config & Constants
# -----------------------------
st.set_page_config(page_title="Freelancer CV Engine", page_icon="🧠", layout="wide")

# Writable path on Streamlit Cloud
DEFAULT_DB = "/mount/tmp/cv_engine.sqlite3" if os.path.isdir("/mount/tmp") else "cv_engine.sqlite3"
DB_PATH = os.environ.get("CV_ENGINE_DB", DEFAULT_DB)

# Admin password: from secrets first, else env var
ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD", os.environ.get("ADMIN_PASSWORD", ""))

DEFAULT_SKILLS_DB = sorted(list({
    # Core
    "project management","excel","power bi","tableau","python","sql","r","visualization",
    "machine learning","statistics","forecasting","regression","classification","nlp",
    # HR / TA
    "talent acquisition","workday","sap successfactors","oracle hcm","saudization","ikva","iktva",
    "compensation","benefits","payroll","workforce planning","hr analytics","people analytics",
    # Energy / O&G
    "aramco","pmc","epcm","feasibility","economics","energy modeling","refinery","gas processing",
    "hydrogen","ccs","ammonia","methanol","renewables","solar","wind","grid",
    # Software/infra
    "git","jira","confluence","linux","docker","kubernetes","api","rest",
}))


# -----------------------------
# DB Helpers & Migration
# -----------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT,
            phone TEXT,
            skills TEXT,
            score REAL,
            filename TEXT,
            text LONGTEXT,
            uploaded_at TEXT,
            birthdate TEXT,
            nationality TEXT,
            oilgas_exp INTEGER,      -- 0/1
            notice_period TEXT,
            work_title TEXT,
            years_experience REAL
        )
        """
    )
    conn.commit()
    _migrate_add_missing_columns(conn, [
        ("birthdate", "TEXT"),
        ("nationality", "TEXT"),
        ("oilgas_exp", "INTEGER"),
        ("notice_period", "TEXT"),
        ("work_title", "TEXT"),
        ("years_experience", "REAL"),
    ])
    return conn


def _migrate_add_missing_columns(conn, columns: List[Tuple[str, str]]):
    """Add columns if they don't exist (safe migration)."""
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(candidates)")
    existing = {row[1] for row in cur.fetchall()}
    for col, ctype in columns:
        if col not in existing:
            cur.execute(f"ALTER TABLE candidates ADD COLUMN {col} {ctype}")
    conn.commit()


def insert_candidate(conn, row: Dict[str, Any]):
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO candidates
           (name,email,phone,skills,score,filename,text,uploaded_at,
            birthdate,nationality,oilgas_exp,notice_period,work_title,years_experience)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            row.get("name"),
            row.get("email"),
            row.get("phone"),
            ", ".join(row.get("skills", [])),
            float(row.get("score", 0.0)) if row.get("score") is not None else None,
            row.get("filename"),
            row.get("text"),
            row.get("uploaded_at", datetime.utcnow().isoformat()),
            row.get("birthdate"),
            row.get("nationality"),
            int(row.get("oilgas_exp")) if row.get("oilgas_exp") is not None else None,
            row.get("notice_period"),
            row.get("work_title"),
            float(row.get("years_experience")) if row.get("years_experience") not in (None, "") else None,
        ),
    )
    conn.commit()


def load_candidates(conn, q: str | None = None) -> pd.DataFrame:
    base = """SELECT id, name, email, phone, skills, score, filename, uploaded_at,
                     birthdate, nationality,
                     oilgas_exp, notice_period, work_title, years_experience
              FROM candidates"""
    params: Tuple[Any, ...] = ()
    if q:
        base += " WHERE name LIKE ? OR email LIKE ? OR phone LIKE ? OR skills LIKE ? OR work_title LIKE ?"
        like = f"%{q}%"
        params = (like, like, like, like, like)
    df = pd.read_sql_query(base, conn, params=params)
    # Pretty map:
    if "oilgas_exp" in df.columns:
        df["oilgas_exp"] = df["oilgas_exp"].map({1: "Yes", 0: "No"}).fillna("")
    return df


def delete_candidate(conn, cand_id: int):
    cur = conn.cursor()
    cur.execute("DELETE FROM candidates WHERE id = ?", (cand_id,))
    conn.commit()


# -----------------------------
# Parsing & Extraction
# -----------------------------
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?:(?:\+\d{1,3}[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?)?\d{3,4}[\s-]?\d{3,4})")
NAME_HINTS = ["name", "candidate", "cv", "curriculum vitae", "resume"]


def read_file(file) -> str:
    suffix = os.path.splitext(file.name)[1].lower()
    if suffix == ".pdf":
        return pdf_extract_text(file)
    elif suffix in (".docx", ".doc"):
        data = file.read()
        return docx2txt.process(io.BytesIO(data))
    else:
        try:
            return file.read().decode("utf-8", errors="ignore")
        except Exception:
            return ""


def normalize_text(t: str) -> str:
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def guess_name(text: str) -> str | None:
    lines = [l.strip() for l in text.splitlines() if l.strip()][:5]
    header = " ".join(lines)
    matches = re.findall(r"\b([A-Z][a-zA-Z'-]+(?:\s+[A-Z][a-zA-Z'-]+){0,3})\b", header)
    if matches:
        candidates = [m for m in matches if m.lower() not in NAME_HINTS]
        if candidates:
            return max(candidates, key=len)
        return max(matches, key=len)
    return None


def extract_contacts(text: str) -> Tuple[str | None, str | None]:
    email = None
    phone = None
    m = EMAIL_RE.search(text)
    if m:
        email = m.group(0)
    for m in PHONE_RE.finditer(text):
        digits = re.sub("\D", "", m.group(0))
        if len(digits) >= 8:
            phone = m.group(0)
            break
    return email, phone


def extract_skills(text: str, skills_db: List[str]) -> List[str]:
    found =
