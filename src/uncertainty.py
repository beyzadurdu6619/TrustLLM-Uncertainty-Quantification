"""
TrustLLM - Uncertainty & Semantic Clustering Module
"""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Anlamsal Vektörleştirme Modeli (Embedding Model)
_model = None


def get_embedding_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def cluster_responses_by_meaning(candidate_responses, threshold=0.75):
    """
    Aday yanıtları anlamsal kosinüs benzerliğine göre kümeleyen fonksiyon.
    Eğer yanıtlar benzerse aynı Küme ID'sini alır.
    """
    if not candidate_responses:
        return []

    model = get_embedding_model()
    embeddings = model.encode(candidate_responses)

    clusters = []
    cluster_mapping = {}
    current_cluster_id = 0

    for i, emb_i in enumerate(embeddings):
        if i in cluster_mapping:
            continue

        cluster_mapping[i] = current_cluster_id

        for j in range(i + 1, len(embeddings)):
            if j in cluster_mapping:
                continue

            emb_j = embeddings[j]
            similarity = cosine_similarity([emb_i], [emb_j])[0][0]

            if similarity >= threshold:
                cluster_mapping[j] = current_cluster_id

        current_cluster_id += 1

    labels = [cluster_mapping[idx] for idx in range(len(candidate_responses))]
    return labels