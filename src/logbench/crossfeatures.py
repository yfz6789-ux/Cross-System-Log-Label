"""Stateless hashing representation used to select diverse source examples."""
import re

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

HEX = re.compile(r"\b(?:0x)?[0-9a-fA-F]{8,}\b")
NUMBER = re.compile(r"\d+")
PERCENT = re.compile(r"%[0-9a-fA-F]{2}")
LONG = re.compile(r"[A-Za-z0-9_\-]{16,}")


def normalise(text):
    value = PERCENT.sub("<PCT>", str(text))
    value = HEX.sub("<HEX>", value)
    value = LONG.sub("<ID>", value)
    return NUMBER.sub("<NUM>", value)


def template(text, depth=3):
    parts = normalise(text).split(" ")
    method = parts[0] if parts else "-"
    uri = parts[1] if len(parts) > 1 else "-"
    path = uri.split("?")[0]
    query = "?Q" if "?" in uri else ""
    segments = [segment for segment in path.split("/") if segment][:depth]
    tail = path.rsplit("/", 1)[-1]
    extension = "." + tail.rsplit(".", 1)[-1][:8] if "." in tail else ""
    traversal = "/DOTDOT" if ".." in path else ""
    return f'{method} /{"/".join(segments)}{extension}{query}{traversal}'


class Representation:
    """Fixed character/word hashing; it fits no vocabulary on target data."""

    def __init__(self, kind="hashing", dimensions=4096, seed=42):
        if kind != "hashing":
            raise ValueError("Only the validated hashing representation is supported")
        self.dimensions = dimensions
        self.seed = seed
        self.fitted = False

    def fit(self, source_texts, target_texts=None):
        half = max(64, self.dimensions // 2)
        self.char = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5), n_features=half,
                                      alternate_sign=False, norm=None, lowercase=True)
        self.word = HashingVectorizer(analyzer="word", ngram_range=(1, 2),
                                      n_features=self.dimensions - half, alternate_sign=False,
                                      norm=None, lowercase=True, token_pattern=r"[^/?&=\s]+")
        self.fitted = True
        return self

    def transform(self, texts):
        if not self.fitted:
            raise RuntimeError("Representation.fit must run before transform")
        texts = [str(text) for text in texts]
        matrix = sp.hstack([self.char.transform(texts), self.word.transform(texts)], format="csr")
        return normalize(matrix).astype(np.float32)


def kcenter(matrix, count, seed=42, candidates=None, pool_cap=20000):
    """Greedy k-center selection over a bounded candidate pool."""
    pool = np.arange(matrix.shape[0]) if candidates is None else np.asarray(candidates, dtype=int)
    count = int(min(count, len(pool)))
    if count <= 0:
        return np.array([], dtype=int)
    rng = np.random.default_rng(seed)
    if len(pool) > pool_cap:
        pool = np.sort(rng.choice(pool, pool_cap, replace=False))
    block = matrix[pool]
    chosen = [int(rng.integers(len(pool)))]
    distance = np.full(len(pool), np.inf)
    for _ in range(count - 1):
        similarity = block @ block[chosen[-1]].T
        similarity = similarity.toarray().ravel() if sp.issparse(similarity) else np.asarray(similarity).ravel()
        distance = np.minimum(distance, 1.0 - similarity)
        distance[chosen] = -np.inf
        best = int(np.argmax(distance))
        if not np.isfinite(distance[best]):
            break
        chosen.append(best)
    return pool[np.asarray(chosen, dtype=int)]
