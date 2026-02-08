import sqlite3
import numpy as np

class EmbeddingDB:
    def __init__(self, path):
        self.conn = sqlite3.connect(path)
        self.cur = self.conn.cursor()

    def get(self, token):
        self.cur.execute(
            "SELECT vector FROM embeddings WHERE word=?",
            (token,)
        )
        row = self.cur.fetchone()
        if row is None:
            raise KeyError(token)
        return np.fromstring(row[0], sep=",", dtype=np.float32)
