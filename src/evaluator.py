import numpy as np

def calculate_bootstrap_ci(y_true, y_pred, num_bootstraps=1000, ci=95):
    """
    Model başarımı için Non-Parametric Bootstrapping ile %95 Güven Aralığı hesaplar.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    bootstrapped_scores = []
    
    np.random.seed(42)
    for _ in range(num_bootstraps):
        indices = np.random.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = np.mean(y_true[indices] == y_pred[indices])
        bootstrapped_scores.append(score)
        
    lower_p = (100 - ci) / 2
    upper_p = 100 - lower_p
    
    lower_bound = np.percentile(bootstrapped_scores, lower_p)
    upper_bound = np.percentile(bootstrapped_scores, upper_p)
    
    return float(lower_bound), float(upper_bound)