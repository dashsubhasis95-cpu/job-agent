import numpy as np

from .config import EMBED_MODEL

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def embed(texts: list[str]) -> list[np.ndarray]:
    vecs = _get_model().encode(texts, normalize_embeddings=True, batch_size=32)
    return [np.asarray(v, dtype=np.float32) for v in vecs]
