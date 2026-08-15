from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

import config


@lru_cache(maxsize=1)
def get_embedding_model():
    return SentenceTransformer(config.EMBEDDING_MODEL)


def embed_texts(texts):
    model = get_embedding_model()
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return np.asarray(embeddings, dtype="float32")
