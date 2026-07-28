import torch
import torch.nn.functional as F

def extract_academic_entity_token(full_generated_text, scores_list, sequence_tokens, nlp_model):
    """
    SpaCy POS tagging kullanarak üretilen jeneratif metinden hedef varlıkları (NOUN/PROPN) izole eder.
    """
    doc = nlp_model(full_generated_text)

    selected_token = ""
    selected_pos = "NONE"
    token_decision_flow = []

    for idx, token in enumerate(doc):
        pos_tag = token.pos_
        word_text = token.text.strip()

        is_valid_pos = False
        decision_status = "REJECTED"
        rationale = ""

        if not word_text.isalpha() or len(word_text) <= 2:
            rationale = "Elendi: Sembol, karakter veya çok kısa token."
        elif pos_tag not in ["NOUN", "PROPN"]:
            rationale = f"Elendi: Dilbilgisel Türü [{pos_tag}] (İsim değil, Sıfat/Fiil/Edat)."
        else:
            is_valid_pos = True
            if not selected_token:
                selected_token = word_text
                selected_pos = pos_tag
                decision_status = "SELECTED"
                rationale = f"✅ SEÇİLDİ: İlk geçen geçerli [{pos_tag}] (İsim/Nesne)."
            else:
                decision_status = "CANDIDATE_SKIPPED"
                rationale = f"Aday: Geçerli [{pos_tag}] ancak daha önceki isim seçildi."

        token_decision_flow.append(
            {
                "step": idx + 1,
                "word": word_text,
                "pos": pos_tag,
                "valid_pos": is_valid_pos,
                "status": decision_status,
                "rationale": rationale,
            }
        )

    if not selected_token and len(doc) > 0:
        for token in doc:
            if token.text.isalpha() and len(token.text) > 2:
                selected_token = token.text
                selected_pos = token.pos_
                break

    word_logit, word_prob = 0.0, 0.0
    if len(scores_list) > 0:
        last_logits = scores_list[-1][0]
        last_id = sequence_tokens[-1].item() if len(sequence_tokens) > 0 else 0
        word_logit = last_logits[last_id].item()
        word_prob = F.softmax(last_logits, dim=-1)[last_id].item()

    return selected_token, selected_pos, token_decision_flow, word_logit, word_prob