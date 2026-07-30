def run_ablation_analysis(doc_input, semantic_entropy: float, is_subj_hybrid: bool):
    """
    Sistemdeki bileşenlerin karara etkisini ölçen Ablation Study.
    """
    is_subj_syntax_only = any(
        token.lemma_.lower() in ["best", "worst", "favorite", "pretty", "tasty"] 
        for token in doc_input
    )
    is_subj_entropy_only = semantic_entropy >= 0.50

    return {
        "syntax_only": is_subj_syntax_only,
        "entropy_only": is_subj_entropy_only,
        "hybrid_full": is_subj_hybrid
    }