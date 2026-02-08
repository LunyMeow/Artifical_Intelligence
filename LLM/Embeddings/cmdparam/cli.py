import argparse
from embedding_db import EmbeddingDB
from system import AutoCommandParamSystem

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--schema", required=True)
    p.add_argument("--embeddings", required=True)
    p.add_argument("--sentence", required=True)
    p.add_argument("--command", required=True)
    args = p.parse_args()

    db = EmbeddingDB(args.embeddings)
    sys = AutoCommandParamSystem(args.schema, db)

    params = sys.extract(args.sentence, args.command)
    print(params)

if __name__ == "__main__":
    main()
