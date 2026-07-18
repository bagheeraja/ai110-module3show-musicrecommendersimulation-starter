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


def apply_ranking_rule(match_scores, dataframe, top_n=5, diversify=True):
    """
    RANKING RULE: Curates, truncates, and sequences the raw data evaluations.
    Uses argpartition to bypass full sorting bottlenecks across 1.2M items.

    When diversify=True, applies an anti-crowding filter that drops a
    candidate if its `artists` or `album_id` already occupies a slot in the
    result set, so near-duplicate tracks (remixes, live cuts, same-album
    songs) don't crowd out the rest of the top N.
    """
    if not diversify:
        # Fast partition array mapping using negative score metrics to grab the largest values
        top_indices = np.argpartition(-match_scores, top_n + 1)[:top_n + 1]

        # Arrange isolated elements sequentially from highest score to lowest
        top_indices = top_indices[np.argsort(-match_scores[top_indices])]
    else:
        # Pull a larger candidate pool since some will be dropped for crowding
        pool_size = min(len(match_scores), max(top_n * 20, top_n + 1))
        pool_indices = np.argpartition(-match_scores, pool_size - 1)[:pool_size]
        pool_indices = pool_indices[np.argsort(-match_scores[pool_indices])]

        seen_artists = set()
        seen_albums = set()
        selected = []
        for idx in pool_indices:
            row = dataframe.iloc[idx]
            artist, album_id = row['artists'], row['album_id']
            if artist in seen_artists or album_id in seen_albums:
                continue
            seen_artists.add(artist)
            seen_albums.add(album_id)
            selected.append(idx)
            if len(selected) == top_n:
                break
        top_indices = np.array(selected)

    # Build structural presentation dataframe payload
    ranked_results = dataframe.iloc[top_indices].copy()
    ranked_results['match_score'] = np.round(match_scores[top_indices], 1)

    return ranked_results[['name', 'artists', 'match_score']]