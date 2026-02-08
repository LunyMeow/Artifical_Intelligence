import json
import argparse
import torch
import torch.nn as nn
from embedding_db import EmbeddingDB


class ParamExtractor(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        return self.fc(x)


class CommandParamSystem:
    def __init__(self, schema_path, embedding_db):
        with open(schema_path, "r", encoding="utf-8") as f:
            self.schema = json.load(f)

        self.embedding_db = embedding_db

        # ✅ EMBEDDING BOYUTUNU GERÇEK BİR KELİMEDEN BUL
        self.embedding_dim = self._detect_embedding_dim()
        self.model = ParamExtractor(self.embedding_dim)

    def _detect_embedding_dim(self):
        """
        DB'de gerçekten var olan bir token bularak
        embedding boyutunu tespit eder
        """
        cursor = self.embedding_db.cur
        cursor.execute("SELECT word FROM embeddings LIMIT 1")
        row = cursor.fetchone()

        if row is None:
            raise RuntimeError("embeddings.db boş")

        vec = self.embedding_db.get(row[0])
        return len(vec)

    def find_param(self, sentence, command):
        tokens = sentence.lower().split()

        if command not in self.schema:
            return {}

        needed = self.schema[command]["params"]
        results = {}

        # Cümledeki tokenların embedding'leri
        token_vecs = {}
        for t in tokens:
            try:
                token_vecs[t] = torch.tensor(
                    self.embedding_db.get(t), dtype=torch.float32
                )
            except KeyError:
                pass

        # Her parametre için en uygun kelimeyi bul
        for param in needed:
            best_token = None
            best_score = -1

            for word, vec in token_vecs.items():
                score = torch.cosine_similarity(
                    self.model(vec), vec, dim=0
                ).item()

                if score > best_score:
                    best_score = score
                    best_token = word

            results[param] = best_token

        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--sentence", required=True)
    parser.add_argument("--command", required=True)

    args = parser.parse_args()

    emb_db = EmbeddingDB(args.embeddings)
    system = CommandParamSystem(args.schema, emb_db)

    params = system.find_param(args.sentence, args.command)
    print(params)


if __name__ == "__main__":
    main()
