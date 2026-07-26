import numpy as np
import torch
import torch.nn.functional as F


def compute_ece(logits, labels, n_bins=10):
    """Calculates Expected Calibration Error (ECE) from raw logits and true
    labels.

    TR: Logit değerlerinden Expected Calibration Error (ECE) skorunu hesaplar.
    """
    probs = F.softmax(logits, dim=1).detach().numpy()
    labels = labels.numpy() if isinstance(labels, torch.Tensor) else labels

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

    return ece


def compute_semantic_entropy(cluster_labels):
    """Computes Semantic Entropy over the probability distribution of response
    clusters.

    TR: Anlamsal kümelerin olasılık dağılımı üzerinden Entropi (Belirsizlik)
    hesaplar.
    """
    _, counts = np.unique(cluster_labels, return_counts=True)
    probabilities = counts / len(cluster_labels)

    semantic_entropy = -np.sum(probabilities * np.log(probabilities + 1e-12))
    return semantic_entropy