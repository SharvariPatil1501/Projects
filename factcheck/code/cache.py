"""Tiny SQLite cache so repeated queries return instantly."""
import hashlib
import json
import sqlite3
import time

DB = "cache.db"


def _conn():
    c = sqlite3.connect(DB)
    c.execute("CREATE TABLE IF NOT EXISTS runs (key TEXT PRIMARY KEY, ts REAL, result TEXT)")
    return c


def _key(prompt: str) -> str:
    return hashlib.sha256(prompt.strip().lower().encode()).hexdigest()


def get(prompt: str, max_age: float = 24 * 3600):
    with _conn() as c:
        row = c.execute("SELECT ts, result FROM runs WHERE key=?", (_key(prompt),)).fetchone()
    if row and time.time() - row[0] < max_age:
        return json.loads(row[1])
    return None


def put(prompt: str, result: dict):
    with _conn() as c:
        c.execute("INSERT OR REPLACE INTO runs VALUES (?,?,?)",
                  (_key(prompt), time.time(), json.dumps(result)))
