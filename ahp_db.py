"""
ahp_db.py — 저장소 계층

Turso(클라우드 SQLite)가 설정되어 있으면 Turso, 아니면 로컬 SQLite 파일을 사용한다.
Streamlit Community Cloud는 재배포 시 로컬 파일이 날아가므로 실서비스는 Turso 권장.

.streamlit/secrets.toml 예시
--------------------------------
TURSO_DATABASE_URL = "libsql://xxxx.turso.io"
TURSO_AUTH_TOKEN   = "eyJhbGciOi..."
ADMIN_PASSWORD     = "..."
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

LOCAL_DB_PATH = os.environ.get("AHP_DB_PATH", "ahp_local.db")

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS surveys (
        sid            TEXT PRIMARY KEY,
        title          TEXT NOT NULL,
        intro          TEXT,
        structure_json TEXT NOT NULL,
        settings_json  TEXT NOT NULL,
        status         TEXT DEFAULT 'open',
        created_at     TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS responses (
        rid          TEXT PRIMARY KEY,
        sid          TEXT NOT NULL,
        respondent   TEXT,
        meta_json    TEXT,
        submitted_at TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS comparisons (
        rid     TEXT NOT NULL,
        sid     TEXT NOT NULL,
        node_id TEXT NOT NULL,
        i       INTEGER NOT NULL,
        j       INTEGER NOT NULL,
        value   REAL NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS node_cr (
        rid     TEXT NOT NULL,
        sid     TEXT NOT NULL,
        node_id TEXT NOT NULL,
        n       INTEGER NOT NULL,
        cr      REAL NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS idx_resp_sid ON responses(sid)",
    "CREATE INDEX IF NOT EXISTS idx_comp_rid ON comparisons(rid)",
    "CREATE INDEX IF NOT EXISTS idx_cr_sid ON node_cr(sid)",
]


# ----------------------------------------------------------------------
# 연결
# ----------------------------------------------------------------------

def _secret(key: str) -> Optional[str]:
    try:
        import streamlit as st  # noqa
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.environ.get(key)


def backend_name() -> str:
    return "Turso (클라우드)" if _secret("TURSO_DATABASE_URL") else f"로컬 SQLite ({LOCAL_DB_PATH})"


def connect():
    url = _secret("TURSO_DATABASE_URL")
    token = _secret("TURSO_AUTH_TOKEN")
    if url:
        try:
            import libsql_experimental as libsql  # type: ignore
            return libsql.connect(database=url, auth_token=token)
        except ImportError:
            pass
        try:
            import libsql  # type: ignore
            return libsql.connect(database=url, auth_token=token)
        except ImportError:
            raise RuntimeError(
                "Turso URL이 설정되어 있으나 libsql 패키지가 없습니다. "
                "`pip install libsql-experimental` 후 다시 실행하세요."
            )
    conn = sqlite3.connect(LOCAL_DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    conn = connect()
    for stmt in SCHEMA:
        conn.execute(stmt)
    conn.commit()


def _rows(cur) -> List[tuple]:
    try:
        return list(cur.fetchall())
    except Exception:
        return []


# ----------------------------------------------------------------------
# 조사(설문) CRUD
# ----------------------------------------------------------------------

def new_id(prefix: str = "") -> str:
    return prefix + uuid.uuid4().hex[:10]


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def create_survey(title: str, intro: str, structure: dict, settings: dict) -> str:
    sid = new_id()
    conn = connect()
    conn.execute(
        "INSERT INTO surveys (sid,title,intro,structure_json,settings_json,status,created_at) "
        "VALUES (?,?,?,?,?,?,?)",
        (sid, title, intro, json.dumps(structure, ensure_ascii=False),
         json.dumps(settings, ensure_ascii=False), "open", now_iso()),
    )
    conn.commit()
    return sid


def update_survey(sid: str, *, title=None, intro=None, structure=None,
                  settings=None, status=None) -> None:
    cur_ = get_survey(sid)
    if not cur_:
        return
    conn = connect()
    conn.execute(
        "UPDATE surveys SET title=?, intro=?, structure_json=?, settings_json=?, status=? WHERE sid=?",
        (
            title if title is not None else cur_["title"],
            intro if intro is not None else cur_["intro"],
            json.dumps(structure if structure is not None else cur_["structure"], ensure_ascii=False),
            json.dumps(settings if settings is not None else cur_["settings"], ensure_ascii=False),
            status if status is not None else cur_["status"],
            sid,
        ),
    )
    conn.commit()


def get_survey(sid: str) -> Optional[Dict[str, Any]]:
    conn = connect()
    cur = conn.execute(
        "SELECT sid,title,intro,structure_json,settings_json,status,created_at "
        "FROM surveys WHERE sid=?", (sid,))
    r = _rows(cur)
    if not r:
        return None
    row = r[0]
    return {
        "sid": row[0], "title": row[1], "intro": row[2] or "",
        "structure": json.loads(row[3]), "settings": json.loads(row[4]),
        "status": row[5], "created_at": row[6],
    }


def list_surveys() -> List[Dict[str, Any]]:
    conn = connect()
    cur = conn.execute(
        "SELECT s.sid, s.title, s.status, s.created_at, "
        "(SELECT COUNT(*) FROM responses r WHERE r.sid = s.sid) "
        "FROM surveys s ORDER BY s.created_at DESC")
    return [{"sid": a, "title": b, "status": c, "created_at": d, "n": e}
            for a, b, c, d, e in _rows(cur)]


def delete_survey(sid: str) -> None:
    conn = connect()
    for t in ("comparisons", "node_cr", "responses", "surveys"):
        conn.execute(f"DELETE FROM {t} WHERE sid=?", (sid,))
    conn.commit()


# ----------------------------------------------------------------------
# 응답 저장/조회
# ----------------------------------------------------------------------

def save_response(sid: str, respondent: str, meta: dict,
                  comparisons: Sequence[tuple], crs: Sequence[tuple]) -> str:
    """
    comparisons: [(node_id, i, j, value), ...]
    crs        : [(node_id, n, cr), ...]
    """
    rid = new_id("r")
    conn = connect()
    conn.execute("INSERT INTO responses (rid,sid,respondent,meta_json,submitted_at) VALUES (?,?,?,?,?)",
                 (rid, sid, respondent or "", json.dumps(meta, ensure_ascii=False), now_iso()))
    for node_id, i, j, v in comparisons:
        conn.execute("INSERT INTO comparisons (rid,sid,node_id,i,j,value) VALUES (?,?,?,?,?,?)",
                     (rid, sid, node_id, int(i), int(j), float(v)))
    for node_id, n, cr in crs:
        conn.execute("INSERT INTO node_cr (rid,sid,node_id,n,cr) VALUES (?,?,?,?,?)",
                     (rid, sid, node_id, int(n), float(cr)))
    conn.commit()
    return rid


def fetch_responses(sid: str) -> List[Dict[str, Any]]:
    conn = connect()
    cur = conn.execute(
        "SELECT rid, respondent, meta_json, submitted_at FROM responses WHERE sid=? ORDER BY submitted_at",
        (sid,))
    return [{"rid": a, "respondent": b, "meta": json.loads(c or "{}"), "submitted_at": d}
            for a, b, c, d in _rows(cur)]


def fetch_comparisons(sid: str) -> List[Dict[str, Any]]:
    conn = connect()
    cur = conn.execute(
        "SELECT rid, node_id, i, j, value FROM comparisons WHERE sid=?", (sid,))
    return [{"rid": a, "node_id": b, "i": c, "j": d, "value": e} for a, b, c, d, e in _rows(cur)]


def fetch_crs(sid: str) -> List[Dict[str, Any]]:
    conn = connect()
    cur = conn.execute("SELECT rid, node_id, n, cr FROM node_cr WHERE sid=?", (sid,))
    return [{"rid": a, "node_id": b, "n": c, "cr": d} for a, b, c, d in _rows(cur)]


def delete_response(rid: str) -> None:
    conn = connect()
    for t in ("comparisons", "node_cr", "responses"):
        conn.execute(f"DELETE FROM {t} WHERE rid=?", (rid,))
    conn.commit()
