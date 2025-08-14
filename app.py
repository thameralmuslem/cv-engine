
"""
Freelancer CV Collector & Analyzer — Streamlit App (Cloud-friendly)
===================================================================

What this app does
------------------
• Upload many CVs (PDF/DOCX/TXT).
• Auto-parse text, detect name/email/phone, extract skills.
• Score each CV against a Job Description (JD) you paste.
• Save everything to a SQLite database (file path is configurable).
• Search, filter, and export to CSV from a simple web UI.

How to run (local)
------------------
pip install -r requirements.txt
streamlit run app.py

Notes
-----
• In Streamlit Cloud, we default DB path to `/mount/tmp/cv_engine.sqlite3` (writable at runtime).
• You can override via env var CV_ENGINE_DB.
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

# Prefer a writable path on Streamlit Cloud
DEFAULT_DB = "/mount/tmp/cv_engine.sqlite3" if os.path.isdir("/mount/tmp") else "cv_engine.sqlite3"
DB_PATH = os.environ.get("CV_ENGINE_DB", DEFAULT_DB)

# A simple, editable skills dictionary (you can expand it in Settings)
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
# Database helpers
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
            uploaded_at TEXT
        )
        """
    )
    conn.commit()
    return conn

def insert_candidate(conn, row: Dict[str, Any]):
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO candidates(name,email,phone,skills,score,filename,text,uploaded_at)
               VALUES (?,?,?,?,?,?,?,?)""",
        (
            row.get("name"),
            row.get("email"),
            row.get("phone"),
            ", ".join(row.get("skills", [])),
            float(row.get("score", 0.0)) if row.get("score") is not None else None,
            row.get("filename"),
            row.get("text"),
            row.get("uploaded_at", datetime.utcnow().isoformat()),
        ),
    )
    conn.commit()

def load_candidates(conn, q: str | None = None) -> pd.DataFrame:
    base = "SELECT id, name, email, phone, skills, score, filename, uploaded_at FROM candidates"
    params: Tuple[Any, ...] = ()
    if q:
        base += " WHERE name LIKE ? OR email LIKE ? OR phone LIKE ? OR skills LIKE ?"
        like = f"%{q}%"
        params = (like, like, like, like)
    df = pd.read_sql_query(base, conn, params=params)
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
    found = []
    low = text.lower()
    for s in skills_db:
        token = s.lower()
        if token in low or token.replace(" ", "-") in low:
            found.append(s)
    return sorted(list(dict.fromkeys(found)))

# -----------------------------
# Scoring
# -----------------------------
def jd_similarity_score(cv_text: str, jd_text: str) -> float:
    if not jd_text or not jd_text.strip():
        return 0.0
    docs = [cv_text, jd_text]
    vec = TfidfVectorizer(stop_words="english", max_features=5000)
    X = vec.fit_transform(docs)
    sim = cosine_similarity(X[0:1], X[1:2])[0, 0]
    return float(np.round(sim * 100.0, 2))

# -----------------------------
# UI: Sidebar Controls
# -----------------------------
def sidebar_controls() -> Dict[str, Any]:
    st.sidebar.header("⚙️ Settings")
    st.sidebar.caption(f"DB Path: `{DB_PATH}`")
    with st.sidebar.expander("Skills Database (click to expand)"):
        skills_text = st.text_area(
            "Edit your skills list (one per line):",
            value="\n".join(st.session_state.get("skills_db", DEFAULT_SKILLS_DB)),
            height=220,
        )
        if st.button("Save Skills DB", use_container_width=True):
            new_db = [s.strip() for s in skills_text.splitlines() if s.strip()]
            st.session_state["skills_db"] = new_db
            st.success("Skills database updated.")
    return {}

# -----------------------------
# UI: Pages
# -----------------------------
def page_upload_analyze(conn):
    st.subheader("📥 Upload & Analyze CVs")

    jd_text = st.text_area("Paste your Job Description (JD) for scoring (optional):", height=160,
                           help="If empty, the app will still parse and index skills but score will be 0.")
    files = st.file_uploader(
        "Upload CV files (PDF/DOCX/TXT). You can select multiple.",
        type=["pdf", "docx", "doc", "txt"],
        accept_multiple_files=True,
    )

    if files and st.button("Analyze & Save", type="primary"):
        processed = 0
        for f in files:
            try:
                raw = read_file(f)
                text = normalize_text(raw)
                name = guess_name(text) or "—"
                email, phone = extract_contacts(text)
                skills = extract_skills(text, st.session_state.get("skills_db", DEFAULT_SKILLS_DB))
                score = jd_similarity_score(text, jd_text)
                row = {
                    "name": name,
                    "email": email or "—",
                    "phone": phone or "—",
                    "skills": skills,
                    "score": score,
                    "filename": f.name,
                    "text": text,
                    "uploaded_at": datetime.utcnow().isoformat(),
                }
                insert_candidate(conn, row)
                processed += 1
            except Exception as e:
                st.warning(f"Failed to parse {f.name}: {e}")
        st.success(f"Processed {processed} CV(s). Go to the Dashboard to view results.")

    st.info("Tip: Update the Skills DB in the sidebar so extraction fits your niche.")

def page_dashboard(conn):
    st.subheader("📊 Dashboard")
    df = load_candidates(conn)
    if df.empty:
        st.info("No candidates yet. Upload some CVs on the Upload page.")
        return

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Candidates", len(df))
    with col2:
        st.metric("Avg Score", f"{df['score'].fillna(0).mean():.1f}")
    with col3:
        st.metric("Top Score", f"{df['score'].fillna(0).max():.1f}")
    with col4:
        st.metric("Skills (unique)", len(set(", ".join(df["skills"].fillna("")).split(", "))) )

    st.divider()
    c1, c2 = st.columns([2,1])
    with c1:
        st.write("### Candidates Table")
        st.dataframe(df.sort_values(by=["score","uploaded_at"], ascending=[False, False]), use_container_width=True)
    with c2:
        st.write("### Quick Actions")
        if st.button("Export CSV", use_container_width=True):
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download candidates.csv", csv, file_name="candidates.csv", mime="text/csv", use_container_width=True)
        cand_id = st.number_input("Delete candidate by ID", min_value=0, step=1, value=0)
        if st.button("Delete", use_container_width=True):
            if cand_id > 0:
                try:
                    delete_candidate(conn, int(cand_id))
                    st.success(f"Deleted candidate ID {cand_id}.")
                except Exception as e:
                    st.error(str(e))
            else:
                st.warning("Enter a valid candidate ID.")

def page_search(conn):
    st.subheader("🔎 Search & Filter")
    q = st.text_input("Search name/email/phone/skills:")
    df = load_candidates(conn, q=q if q else None)

    min_score, max_score = st.slider("Score range", 0.0, 100.0, (0.0, 100.0))
    df = df[(df["score"].fillna(0) >= min_score) & (df["score"].fillna(0) <= max_score)]

    st.dataframe(df.sort_values(by=["score","uploaded_at"], ascending=[False, False]), use_container_width=True)

    if not df.empty:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download filtered CSV", csv, file_name="filtered_candidates.csv", mime="text/csv")

# -----------------------------
# App Entry
# -----------------------------
def main():
    if "skills_db" not in st.session_state:
        st.session_state["skills_db"] = DEFAULT_SKILLS_DB

    conn = init_db()

    st.title("🧠 Freelancer CV Collector & Analyzer")
    st.caption("Upload → Parse → Score → Search. Private, simple, and fast.")

    sidebar_controls()

    tab1, tab2, tab3 = st.tabs(["Upload & Analyze", "Dashboard", "Search & Export"])
    with tab1:
        page_upload_analyze(conn)
    with tab2:
        page_dashboard(conn)
    with tab3:
        page_search(conn)

    st.markdown("---")
    st.caption("Tip: Deploy to Streamlit Cloud to get a public URL where candidates can upload directly.")
    st.caption("If you see DB write errors on Streamlit Cloud, set CV_ENGINE_DB to /mount/tmp/cv_engine.sqlite3 in app settings.")

if __name__ == "__main__":
    main()
