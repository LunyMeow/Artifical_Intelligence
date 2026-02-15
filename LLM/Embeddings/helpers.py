import re
import math
import random

# =========================
# AKTİVASYON FONKSİYONLARI
# =========================
exp = math.exp
log = math.log
tanh = math.tanh

# =========================
# TOKENIZER MODE
# =========================
class TokenizerMode:
    WORD = "word"
    SUBWORD = "subword"
    BPE = "bpe"

# =========================
# EMB_SIZE and EMB_RANGE
# =========================
EMB_SIZE = 50
EMB_RANGE = range(EMB_SIZE)

# =========================
# ÖZEL TOKEN SETİ (Merkezi tanım)
# Bu token'lar hem createEmbeddings.py hem commandCreator.py hem de
# Buildv1_3_2.cpp tarafından kullanılır. Küçük harf ile normalize edilmiş
# hali DB'de tek bir kayıt olarak yaşar — <DIR> ve <dir> artık ayrı vektör
# oluşturmaz.
# =========================
COMMAND_TOKENS = {
    "<dir>", "<file>", "<path>", "<number>", "<ip>", "<url>", "<var>", "<end>"
}

# Bilinen komutlar (normalize edilmeden geçirilir)
KNOWN_COMMANDS = {
    "mkdir", "rm", "cd", "ls", "cp", "mv", "touch", "cat",
    "grep", "find", "chmod", "chown", "sudo", "apt", "git",
    "python", "python3", "node", "npm", "bash", "sh", "echo",
    "tar", "zip", "unzip", "wget", "curl", "ssh", "scp"
}

class helpers:
    def normalize_command_params(tokens):
        """
        Komut parametrelerini tip placeholder'larına dönüştürür.
        createEmbeddings.py ile TAMAMEN AYNI mantık — tek kaynak.

        DÜZELTME: Tüm placeholder'lar küçük harf (<dir>, <file> vb.)
        Böylece unified DB'de çift kayıt (ör. <DIR> + <dir>) oluşmaz.
        """
        normalized = []

        for token in tokens:
            # Zaten bir COMMAND_TOKEN ise dokunma
            if token in COMMAND_TOKENS:
                normalized.append(token)
                continue

            if token in KNOWN_COMMANDS:
                normalized.append(token)
                continue

            # Bayraklar (-x, --verbose vb.) olduğu gibi
            if token.startswith("-"):
                normalized.append(token)
                continue

            # Parametre tipi tespiti
            if re.match(r"^[/\\]|.*[/\\].*", token):
                normalized.append("<path>")
            elif re.match(r".*\.[a-zA-Z0-9]+$", token):
                normalized.append("<file>")
            elif re.match(r"^\d+$", token):
                normalized.append("<number>")
            elif re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", token):
                normalized.append("<ip>")
            elif re.match(r"^https?://", token):
                normalized.append("<url>")
            elif re.match(r"^\$[A-Z_]+$", token):
                normalized.append("<var>")
            else:
                normalized.append("<dir>")

        return normalized

    def get_activation_info(activation_type):
        """
        Aktivasyon fonksiyonunun çıktı aralığını döndürür.
        Returns: (min_val, max_val, activation_function)
        """
        activations = {
            "sigmoid":    (0.0, 1.0,            lambda x: 1.0 / (1.0 + exp(-x))),
            "tanh":       (-1.0, 1.0,            lambda x: tanh(x)),
            "relu":       (0.0, float('inf'),    lambda x: max(0.0, x)),
            "leaky_relu": (float('-inf'), float('inf'), lambda x: x if x > 0.0 else 0.01 * x),
            "elu":        (-0.01, float('inf'),  lambda x: x if x >= 0.0 else 0.01 * (exp(x) - 1.0)),
            "softplus":   (0.0, float('inf'),    lambda x: log(1.0 + exp(x))),
            "linear":     (float('-inf'), float('inf'), lambda x: x)
        }
        return activations.get(activation_type, activations["sigmoid"])

    def normalize_vector(vec, min_val, max_val):
        """
        Vektörü aktivasyon fonksiyonunun çıktı aralığına göre normalize eder.
        """
        vec_min = min(vec)
        vec_max = max(vec)

        if vec_max == vec_min:
            normalized = [(min_val + max_val) / 2.0] * len(vec)
        else:
            normalized = [(v - vec_min) / (vec_max - vec_min) for v in vec]

            if min_val != float('-inf') and max_val != float('inf'):
                normalized = [min_val + v * (max_val - min_val) for v in normalized]
            elif max_val != float('inf'):
                normalized = [min_val + v * max_val for v in normalized]
            elif min_val != float('-inf'):
                normalized = [min_val + v * abs(min_val) for v in normalized]

        return normalized

    def tokenize(
        sentence,
        is_command=False,
        mode=TokenizerMode.WORD,
        subword_n=3,
        bpe_tokenizer=None
    ):
        """
        Cümleyi belirtilen moda göre tokenize eder.
        is_command=True iken normalize_command_params çalışır.
        Tüm token'lar lower() geçirilerek DB ile tutarlı kalır.
        """
        sentence = sentence.lower()

        # =========================
        # BPE TOKENIZER
        # =========================
        if mode == TokenizerMode.BPE:
            if not bpe_tokenizer:
                raise ValueError("BPE tokenizer verilmedi ancak BPE modu seçildi")

            tokens = []
            words = sentence.split()

            if is_command:
                words = helpers.normalize_command_params(words)

            for w in words:
                if w in COMMAND_TOKENS:
                    tokens.append(w)
                    continue

                try:
                    ids = bpe_tokenizer.encode(w)
                    for token_id in ids:
                        if token_id in bpe_tokenizer.id_to_token:
                            token_bytes = bpe_tokenizer.id_to_token[token_id]
                            token_str = token_bytes.decode("utf-8", errors="ignore")
                            tokens.append(token_str)
                except Exception as e:
                    print(f"[WARNING] BPE encode hatası '{w}': {e}")
                    tokens.append(w)

            return tokens

        # =========================
        # WORD TOKENIZER
        # =========================
        if mode == TokenizerMode.WORD:
            tokens = sentence.split()
            if is_command:
                tokens = helpers.normalize_command_params(tokens)
            return tokens

        # =========================
        # SUBWORD TOKENIZER (char n-gram)
        # =========================
        if mode == TokenizerMode.SUBWORD:
            tokens = []
            words = sentence.split()

            if is_command:
                words = helpers.normalize_command_params(words)

            for w in words:
                if w in COMMAND_TOKENS:
                    tokens.append(w)
                    continue

                L = len(w)
                if L < subword_n:
                    tokens.append(w)
                    continue

                for i in range(L - subword_n + 1):
                    tokens.append(w[i:i + subword_n])

            return tokens

        return []

    def sentence_embedding(
        sentence,
        embeddings,
        act_min,
        act_max,
        is_command=False,
        tokenizer_mode=TokenizerMode.WORD,
        subword_n=3,
        bpe_tokenizer=None
    ):
        """
        Cümle gömülüşünü hesaplar.

        DÜZELTME: Artık tek bir 'embeddings' dict'i alır (unified DB).
        is_command=True olduğunda token'lar normalize edilir ama
        lookup aynı dict üzerinden yapılır.
        """
        tokens = helpers.tokenize(
            sentence,
            is_command=is_command,
            mode=tokenizer_mode,
            subword_n=subword_n,
            bpe_tokenizer=bpe_tokenizer
        )

        vec = [0.0] * EMB_SIZE
        cnt = 0

        for w in tokens:
            if w in embeddings:
                vw = embeddings[w]
                for i in EMB_RANGE:
                    vec[i] += vw[i]
                cnt += 1

        if cnt:
            inv = 1.0 / cnt
            for i in EMB_RANGE:
                vec[i] *= inv

        vec = helpers.normalize_vector(vec, act_min, act_max)
        return vec