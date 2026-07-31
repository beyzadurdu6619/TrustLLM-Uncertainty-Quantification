# src/explanation.py

def generate_refusal_explanation(prompt_text, refusal_reason, extracted_entity=""):
    """
    Model bir yanıtı reddettiğinde veya düşük güven verdiğinde 
    kullanıcıya yapıcı açıklama ve doğrulama kaynakları üretir.
    """
    explanations = {
        "SUBJECTIVE": {
            "title": "🎯 Öznellik Sebebiyle Yanıt Kısıtlandı",
            "message": f"'{prompt_text}' sorgusu kişisel tercihlere ve göreceli değerlendirmelere dayanmaktadır.",
            "suggestion": "Nesnel bir yanıt almak için sorguyu spesifik kriterlere (örneğin: 'IMDb puanı en yüksek film') göre daraltabilirsiniz."
        },
        "LOW_CONFIDENCE": {
            "title": "📉 Bilgi Yetersizliği / Yüksek Belirsizlik",
            "message": f"Model, '{prompt_text}' konusu hakkında yeterince kararlı bir bilgiye sahip değil (Anlamsal Entropi Yüksek).",
            "suggestion": f"Verilen yanıtın aşağıdaki kaynaklardan veya resmi dokümantasyonlardan doğrulanması önerilir:\n"
                          f"• Wikipedia / Google Scholar üzerinde '{extracted_entity or prompt_text}' araması yapın.\n"
                          f"• Konuyla ilgili alan uzmanı kaynaklara başvurun."
        },
        "AMBIGUOUS": {
            "title": "❓ Muğlak Sorgu Bütünlüğü",
            "message": "Sorgu birden fazla farklı anlama gelebilecek eksik bağlam içeriyor olabilir.",
            "suggestion": "Lütfen sorguya zaman, konum veya kategori bilgisi ekleyerek tekrar deneyin."
        }
    }
    
    return explanations.get(refusal_reason, explanations["LOW_CONFIDENCE"])