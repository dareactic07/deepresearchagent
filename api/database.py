import sqlite3
import os
from datetime import datetime

# Local SQLite file
DB_PATH = "chat_history.db"

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                topic TEXT,
                created_at TIMESTAMP
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                content TEXT,
                timestamp TIMESTAMP,
                FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
            )
        ''')

def create_session(session_id: str, topic: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('INSERT INTO sessions (id, topic, created_at) VALUES (?, ?, ?)', 
                    (session_id, topic, datetime.utcnow()))

def add_message(session_id: str, role: str, content: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)',
                    (session_id, role, content, datetime.utcnow()))

def get_sessions():
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute('SELECT * FROM sessions ORDER BY created_at DESC')
        return [dict(row) for row in cur.fetchall()]

def get_messages(session_id: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute('SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp ASC', (session_id,))
        return [dict(row) for row in cur.fetchall()]

def delete_session(session_id: str):
    with sqlite3.connect(DB_PATH) as conn:
        # SQLite PRAGMA foreign_keys = ON; needs to be set per connection for cascade to work natively
        conn.execute('PRAGMA foreign_keys = ON;')
        conn.execute('DELETE FROM sessions WHERE id = ?', (session_id,))
