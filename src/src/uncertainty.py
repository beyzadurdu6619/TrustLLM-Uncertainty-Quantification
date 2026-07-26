from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import torch
import torch.nn.functional as F

# Pre-load embedding model for semantic clustering
# Anlamsal kümeleme için embedding modelini yüklüyoruz
_embedder = SentenceTransformer("all-MiniLM-L6-v2")


def enable_dropout(model):
    """Forces all Dropout layers in the model to remain active during
    evaluation."""
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()


def estimate_mc_uncertainty(model, x_input, num_samples=20):
    """Computes mean probability and epistemic uncertainty (variance) using Monte
    Carlo Dropout.

    TR: MC Dropout kullanarak ortalama olasılık ve Epistemic belirsizlik
    (varyans) hesaplar.
    """
    model.eval()
    enable_dropout(model)

    mc_predictions = []

    with torch.no_grad():
        for _ in range(num_samples):
            logits = model(x_input)
            probs = F.softmax(logits, dim=-1)
            mc_predictions.append(probs)

    mc_predictions = torch.stack(mc_predictions)
    mean_probs = torch.mean(mc_predictions, dim=0)
    epistemic_uncertainty = torch.var(mc_predictions, dim=0)

    return mean_probs, epistemic_uncertainty


def cluster_responses_by_meaning(responses, distance_threshold=0.3):
    """Clusters generated responses based on semantic similarity using
    hierarchical clustering.

    TR: Üretilen metin yanıtlarını anlamsal benzerliklerine göre gruplar.
    """
    embeddings = _embedder.encode(responses)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=distance_threshold,
    )

    cluster_labels = clustering.fit_predict(embeddings)
    return cluster_labels