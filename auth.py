
import sqlite3
import bcrypt
import re
from pathlib import Path

DB_PATH = Path(__file__).parent / "datagpt.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


def create_table():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL COLLATE NOCASE UNIQUE,
                email TEXT NOT NULL COLLATE NOCASE UNIQUE,
                password TEXT NOT NULL
            )
        """)


def validate_email(email):
    pattern = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
    return re.fullmatch(pattern, email.strip()) is not None


def validate_password(password):

    if len(password) < 8:
        return False, "Password must have at least 8 characters."

    if not re.search(r"[A-Z]", password):
        return False, "Add at least one uppercase letter."

    if not re.search(r"[a-z]", password):
        return False, "Add at least one lowercase letter."

    if not re.search(r"[0-9]", password):
        return False, "Add at least one number."

    if not re.search(r"[^A-Za-z0-9]", password):
        return False, "Add at least one special character."

    if len(password.encode("utf-8")) > 72:
        return False, "Password is too long. Use up to 72 UTF-8 bytes."

    return True, "Password is valid."


def signup(username, email, password):

    username = username.strip()
    email = email.strip().lower()

    if not username:
        return False, "Username is required."

    if not validate_email(email):
        return False, "Enter a valid email address."

    valid, message = validate_password(password)

    if not valid:
        return False, message

    try:
        with get_connection() as conn:

            existing = conn.execute("""
                SELECT id FROM users
                WHERE username = ? COLLATE NOCASE
                   OR email = ? COLLATE NOCASE
            """, (username, email)).fetchone()

            if existing:
                return False, "Username or email already exists."

            hashed_password = bcrypt.hashpw(
                password.encode("utf-8"),
                bcrypt.gensalt()
            )

            conn.execute("""
                INSERT INTO users (username, email, password)
                VALUES (?, ?, ?)
            """, (username, email, hashed_password))

        return True, "Account created successfully!"

    except sqlite3.IntegrityError:
        return False, "Username or email already exists."


def login(email, password):

    email = email.strip().lower()

    if not validate_email(email):
        return False, None

    with get_connection() as conn:

        user = conn.execute("""
            SELECT id, username, password
            FROM users
            WHERE email = ? COLLATE NOCASE
        """, (email,)).fetchone()

    if user:

        stored_password = user[2]

        if bcrypt.checkpw(
            password.encode("utf-8"),
            stored_password
        ):
            return True, {
                "id": user[0],
                "username": user[1]
            }

    return False, None
