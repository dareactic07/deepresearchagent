def calculate_score(relevance: float, clarity: float, agreement: float = 1.0) -> float:
    """
    Calculate an overall confidence score based on relevance, clarity, and source agreement.
    All inputs should be between 0.0 and 1.0.
    """
    score = (relevance * 0.5) + (clarity * 0.3) + (agreement * 0.2)
    return round(max(0.0, min(1.0, score)), 2)
