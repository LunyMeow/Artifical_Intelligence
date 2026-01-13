from collections import Counter
import json
from math import exp, tanh, log


class ByteBPETokenizer:
    def __init__(self, vocab_size=16000):
        self.vocab_size = vocab_size

        self.token_to_id = {bytes([i]): i for i in range(256)}
        self.id_to_token = {i: bytes([i]) for i in range(256)}

        self.merges = []

    # --------------------------------------------------
    # ACTIVATION INFO
    # --------------------------------------------------
    @staticmethod
    def get_activation_info(activation_type):
        activations = {
            "sigmoid": (0.0, 1.0, lambda x: 1.0 / (1.0 + exp(-x))),
            "tanh": (-1.0, 1.0, lambda x: tanh(x)),
            "relu": (0.0, float('inf'), lambda x: max(0.0, x)),
            "leaky_relu": (float('-inf'), float('inf'), lambda x: x if x > 0.0 else 0.01 * x),
            "elu": (-0.01, float('inf'), lambda x: x if x >= 0.0 else 0.01 * (exp(x) - 1.0)),
            "softplus": (0.0, float('inf'), lambda x: log(1.0 + exp(x))),
            "linear": (float('-inf'), float('inf'), lambda x: x)
        }
        return activations.get(activation_type, activations["sigmoid"])

    # --------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------
    @staticmethod
    def normalize_token_id(token_id, vocab_size, activation_type):
        min_val, max_val, _ = ByteBPETokenizer.get_activation_info(activation_type)

        x = token_id / (vocab_size - 1)

        if min_val == float('-inf') or max_val == float('inf'):
            return (x - 0.5) * 2.0
        else:
            return min_val + x * (max_val - min_val)

    @staticmethod
    def denormalize_token_value(value, vocab_size, activation_type):
        min_val, max_val, _ = ByteBPETokenizer.get_activation_info(activation_type)

        if min_val == float('-inf') or max_val == float('inf'):
            x = (value / 2.0) + 0.5
        else:
            x = (value - min_val) / (max_val - min_val)

        token_id = int(round(x * (vocab_size - 1)))
        return max(0, min(vocab_size - 1, token_id))

    @staticmethod
    def normalize_ids(ids, vocab_size, activation_type):
        return [
            ByteBPETokenizer.normalize_token_id(i, vocab_size, activation_type)
            for i in ids
        ]

    @staticmethod
    def denormalize_ids(values, vocab_size, activation_type):
        return [
            ByteBPETokenizer.denormalize_token_value(v, vocab_size, activation_type)
            for v in values
        ]

    # --------------------------------------------------
    # BYTE ENCODE
    # --------------------------------------------------
    @staticmethod
    def _byte_encode(text: str):
        return [bytes([b]) for b in text.encode("utf-8")]

    # --------------------------------------------------
    # PAIR COUNT
    # --------------------------------------------------
    @staticmethod
    def _get_pair_counts(sequences):
        pairs = Counter()
        for seq in sequences:
            pairs.update(zip(seq, seq[1:]))
        return pairs

    # --------------------------------------------------
    # MERGE PAIR
    # --------------------------------------------------
    @staticmethod
    def _merge_pair(pair, sequences):
        a, b = pair
        ab = a + b
        new_sequences = []

        for seq in sequences:
            new_seq = []
            i = 0
            L = len(seq)

            while i < L:
                if i + 1 < L and seq[i] == a and seq[i + 1] == b:
                    new_seq.append(ab)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1

            new_sequences.append(new_seq)

        return new_sequences, ab

    # --------------------------------------------------
    # TRAIN
    # --------------------------------------------------
    def train(self, texts):
        sequences = [self._byte_encode(t) for t in texts]
        next_id = len(self.token_to_id)

        while len(self.token_to_id) < self.vocab_size:
            pair_counts = self._get_pair_counts(sequences)
            if not pair_counts:
                break

            best_pair = pair_counts.most_common(1)[0][0]
            sequences, new_token = self._merge_pair(best_pair, sequences)

            if new_token in self.token_to_id:
                continue

            self.token_to_id[new_token] = next_id
            self.id_to_token[next_id] = new_token
            self.merges.append(best_pair)
            next_id += 1

        return len(self.token_to_id)

    # --------------------------------------------------
    # ENCODE
    # --------------------------------------------------
    def encode(self, text: str):
        seq = self._byte_encode(text)

        for a, b in self.merges:
            new_seq = []
            i = 0
            L = len(seq)

            while i < L:
                if i + 1 < L and seq[i] == a and seq[i + 1] == b:
                    new_seq.append(a + b)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1

            seq = new_seq

        return [self.token_to_id[token] for token in seq]

    # --------------------------------------------------
    # DECODE
    # --------------------------------------------------
    def decode(self, ids):
        return b"".join(self.id_to_token[i] for i in ids).decode(
            "utf-8", errors="replace"
        )

    # --------------------------------------------------
    # SAVE / LOAD
    # --------------------------------------------------
    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "vocab": {k.hex(): v for k, v in self.token_to_id.items()},
                    "merges": [(a.hex(), b.hex()) for a, b in self.merges],
                },
                f,
            )

    def load(self, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        self.token_to_id = {bytes.fromhex(k): v for k, v in data["vocab"].items()}
        self.id_to_token = {v: bytes.fromhex(k) for k, v in data["vocab"].items()}
        self.merges = [(bytes.fromhex(a), bytes.fromhex(b)) for a, b in data["merges"]]



if __name__ == "__main__":
    import time
    corpus = []

    with open("sentences.txt", "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    words = [word for line in lines for word in line.split()]

    corpus = words


    tokenizer = ByteBPETokenizer(vocab_size=500)

    t0 = time.time()
    vocab_size = tokenizer.train(corpus)
    t1 = time.time()

    print(f"Training vocab size: {vocab_size}")
    print(f"Training time: {t1 - t0:.4f}s")

    for i in ["dizini","dizine","ghkgkfgyı"]:
        print("-"*10)
        text = i

        ids = tokenizer.encode(text)
        normIds = tokenizer.normalize_ids(ids,tokenizer.vocab_size,"tanh")
        deNormIds = tokenizer.denormalize_ids(normIds,tokenizer.vocab_size,"tanh")
        decoded = tokenizer.decode(deNormIds)

        print("Input      :", text)
        print("IDs        :", ids)
        print("Decoded    :", decoded)
