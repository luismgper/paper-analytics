import sqlite3
from typing import Any, List, Tuple, Optional


class SQLite:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self.cursor: Optional[sqlite3.Cursor] = None

    def connect(self):
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None

    def execute(self, query: str, params: Tuple = ()) -> None:
        if not self.cursor:
            raise RuntimeError("Database not connected.")
        self.cursor.execute(query, params)
        self.conn.commit()

    
