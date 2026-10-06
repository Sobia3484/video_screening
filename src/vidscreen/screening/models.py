"""Text classifiers for relevant (1) vs irrelevant (0).

* tfidf : word + character n-gram TF-IDF -> logistic regression (robust for small data, Roman Urdu, typos)
* embed : multilingual sentence embeddings (transformer encoder) -> logistic regression
Hybrid variants multiply the probability by the rule gate (a strong CPR/choking/AED/... term must be present).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def make_estimator(kind: str, seed: int = 42):
    from sklearn.linear_model import LogisticRegression

    if kind == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import FeatureUnion, Pipeline

        feats = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)),
        ])
        return Pipeline([("features", feats),
                         ("lr", LogisticRegression(C=4.0, class_weight="balanced", max_iter=3000, random_state=seed))])
    if kind == "embed":
        return LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000, random_state=seed)
    raise ValueError(kind)


def encode_texts(texts, cache_dir: Path, model_name: str = EMBEDDING_MODEL, batch_size: int = 32) -> np.ndarray:
    """Sentence embeddings for all texts, cached on disk (the model is downloaded once from Hugging Face)."""
    texts = [t[:1000] for t in texts]
    key = hashlib.sha1((model_name + "\n" + "\n".join(texts)).encode("utf-8")).hexdigest()[:16]
    path = Path(cache_dir) / f"embeddings_{key}.npy"
    if path.exists():
        return np.load(path)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    emb = model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)
    return emb
