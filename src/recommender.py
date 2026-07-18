import numpy as np
import pandas as pd

def apply_scoring_rule(seed_vector, matrix):
    """
    SCORING RULE: Evaluates a standalone candidate track relative to user preferences.
    Uses vectorized NumPy broadcasting to calculate multi-dimensional Euclidean distance.
    """
    delta = matrix - seed_vector
    distances = np.linalg.norm(delta, axis=1)
    
    # Transform raw distances into a clear proximity score bound between 0-100%
    match_scores = 1 / (1 + distances) * 100
    return match_scores


def apply_ranking_rule(match_scores, dataframe, top_n=5):
    """
    RANKING RULE: Curates, truncates, and sequences the raw data evaluations.
    Uses argpartition to bypass full sorting bottlenecks across 1.2M items.
    """
    # Fast partition array mapping using negative score metrics to grab the largest values
    top_indices = np.argpartition(-match_scores, top_n + 1)[:top_n + 1]
    
    # Arrange isolated elements sequentially from highest score to lowest
    top_indices = top_indices[np.argsort(-match_scores[top_indices])]
    
    # Build structural presentation dataframe payload
    ranked_results = dataframe.iloc[top_indices].copy()
    ranked_results['match_score'] = np.round(match_scores[top_indices], 1)
    
    return ranked_results[['name', 'artists', 'match_score']]