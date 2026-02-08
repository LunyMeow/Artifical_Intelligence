import json
import torch
import numpy as np

class AutoCommandParamSystem:
    def __init__(self, schema_path, embedding_db):
        with open(schema_path, "r", encoding="utf-8") as f:
            self.schema = json.load(f)
        self.db = embedding_db
        self.dim = self._detect_dim()

    def _detect_dim(self):
        self.db.cur.execute("SELECT word FROM embeddings LIMIT 1")
        token = self.db.cur.fetchone()[0]
        return len(self.db.get(token))

    def _cos(self, a, b):
        return torch.cosine_similarity(a, b, dim=0).item()

    def extract(self, sentence, command):
        tokens = sentence.lower().split()
        param_count = len(self.schema[command]["params"])

        # Komut embedding'i (varsa)
        try:
            cmd_vec = torch.tensor(self.db.get(command))
        except:
            cmd_vec = None

        scored = []

        for t in tokens:
            try:
                v = torch.tensor(self.db.get(t))
                score = self._cos(v, cmd_vec) if cmd_vec is not None else 0.0
                scored.append((t, score))
            except KeyError:
                # DB'de YOK → EN GÜÇLÜ PARAMETRE ADAYI
                scored.append((t, -1.0))

        # EN DÜŞÜK SKOR = EN FAZLA BİLGİ
        scored.sort(key=lambda x: x[1])

        return scored[:param_count]
