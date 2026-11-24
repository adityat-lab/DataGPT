
import sqlite3
import json
import hashlib
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).parent / "datagpt.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


def create_history_tables():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS dataset_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                file_hash TEXT NOT NULL,
                filename TEXT NOT NULL,
                rows_count INTEGER NOT NULL,
                columns_count INTEGER NOT NULL,
                columns_json TEXT,
                uploaded_at TEXT NOT NULL,
                UNIQUE(user_id, file_hash)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS analysis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                dataset_id INTEGER,
                analysis_type TEXT NOT NULL,
                analysis_text TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)


def calculate_file_hash(file_bytes):
    return hashlib.sha256(file_bytes).hexdigest()


def save_dataset(user_id, file_bytes, filename, df):
    file_hash = calculate_file_hash(file_bytes)

    with get_connection() as conn:
        existing = conn.execute("""
            SELECT id FROM dataset_history
            WHERE user_id = ? AND file_hash = ?
        """, (user_id, file_hash)).fetchone()

        if existing:
            return existing[0]

        cursor = conn.execute("""
            INSERT INTO dataset_history (
                user_id, file_hash, filename,
                rows_count, columns_count,
                columns_json, uploaded_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            file_hash,
            filename,
            len(df),
            len(df.columns),
            json.dumps([str(col) for col in df.columns]),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))

        return cursor.lastrowid


def save_analysis(user_id, dataset_id, analysis_type, analysis_text):
    if not analysis_text:
        return

    with get_connection() as conn:
        conn.execute("""
            INSERT INTO analysis_history (
                user_id, dataset_id, analysis_type,
                analysis_text, created_at
            )
            VALUES (?, ?, ?, ?, ?)
        """, (
            user_id,
            dataset_id,
            analysis_type,
            str(analysis_text),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))


def get_user_datasets(user_id):
    with get_connection() as conn:
        return conn.execute("""
            SELECT id, filename, rows_count, columns_count,
                   uploaded_at, columns_json
            FROM dataset_history
            WHERE user_id = ?
            ORDER BY id DESC
        """, (user_id,)).fetchall()


def get_user_analyses(user_id):
    with get_connection() as conn:
        return conn.execute("""
            SELECT id, dataset_id, analysis_type,
                   analysis_text, created_at
            FROM analysis_history
            WHERE user_id = ?
            ORDER BY id DESC
        """, (user_id,)).fetchall()


def delete_dataset(user_id, dataset_id):
    with get_connection() as conn:
        conn.execute("""
            DELETE FROM analysis_history
            WHERE user_id = ? AND dataset_id = ?
        """, (user_id, dataset_id))

        conn.execute("""
            DELETE FROM dataset_history
            WHERE user_id = ? AND id = ?
        """, (user_id, dataset_id))
