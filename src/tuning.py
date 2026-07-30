def compute_adaptive_tuning(user_prompt: str, nlp_model):
    """
    Sorgudaki nesnel/bilimsel niyetleri analiz ederek Otomatik Eşik (threshold)
    ve Sıcaklık (temperature) değerlerini hesaplar.
    """
    doc_input = nlp_model(user_prompt)
    has_fact_anchor = any(
        token.lemma_.lower() in [
            "capital", "country", "city", "mountain", "river", 
            "method", "formula", "element", "star", "planet"
        ] 
        or token.pos_ in ["PROPN"] 
        for token in doc_input
    )

    base_threshold = 0.55
    base_temperature = 1.50

    if has_fact_anchor:
        adaptive_threshold = 0.75
        adaptive_temperature = 0.30
        thresh_reason = "Sorguda nesnel/bilimsel bir çapa (fact anchor) tespit edildi. Yanıtın maskelenmesini önleyip nihai cevabı almak için eşik değeri yükseltildi."
        temp_reason = "Sorgunun tek bir doğru cevabı olduğu için modelin rastgele kelime uydurmasını (hallucination) engellemek amacıyla sıcaklık düşürüldü."
    else:
        adaptive_threshold = 0.45
        adaptive_temperature = 0.80
        thresh_reason = "Sorgu öznel veya ucu açık görünüyor. Güvenlik ağını sıkılaştırmak ve halüsinasyonu engellemek için eşik düşürüldü."
        temp_reason = "Öznel sorgularda modelin anlamsal varyansını (entropisini) tam ölçebilmek için sıcaklık standart seviyede tutuldu."

    return {
        "adaptive_threshold": adaptive_threshold,
        "adaptive_temperature": adaptive_temperature,
        "base_threshold": base_threshold,
        "base_temperature": base_temperature,
        "thresh_reason": thresh_reason,
        "temp_reason": temp_reason,
        "doc_input": doc_input
    }