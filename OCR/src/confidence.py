def combine(ocr, validation, context, consistency=0.5):
    # Weighted, interpretable confidence model.
    score = 0.50*ocr + 0.20*validation + 0.15*context + 0.15*consistency
    return round(max(0.0, min(1.0, score)), 3)


def label(score):
    if score >= 0.85: return 'high'
    if score >= 0.70: return 'medium'
    return 'low'
