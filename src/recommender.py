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


def explain_match(seed_vector, track_vector, feature_cols, feature_ranges, top_k=3):
    """
    Builds a short human-readable reason string for why a track matched,
    naming the `top_k` features where the track is closest to the seed.

    Diffs are normalized by each feature's catalog-wide range before
    comparing, since raw absolute differences would unfairly favor
    naturally small-range features like `key` (0-11) or `mode` (0/1) over
    0-1 continuous features like `valence` or `energy`.
    """
    per_feature_diff = np.abs(track_vector - seed_vector) / feature_ranges
    closest_indices = np.argsort(per_feature_diff)[:top_k]
    closest_features = [feature_cols[i] for i in closest_indices]
    return "Closely matches on " + ", ".join(closest_features)


def apply_ranking_rule(match_scores, dataframe, top_n=5, diversify=True,
                        seed_vector=None, feature_matrix=None, feature_cols=None):
    """
    RANKING RULE: Curates, truncates, and sequences the raw data evaluations.
    Uses argpartition to bypass full sorting bottlenecks across 1.2M items.

    When diversify=True, applies an anti-crowding filter that drops a
    candidate if its `artists` or `album_id` already occupies a slot in the
    result set, so near-duplicate tracks (remixes, live cuts, same-album
    songs) don't crowd out the rest of the top N.

    When seed_vector, feature_matrix, and feature_cols are all provided, an
    extra `reasons` column is added, explaining each result via
    explain_match().
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

    columns = ['name', 'artists', 'match_score']
    if seed_vector is not None and feature_matrix is not None and feature_cols is not None:
        feature_ranges = np.ptp(feature_matrix, axis=0)
        feature_ranges[feature_ranges == 0] = 1  # avoid divide-by-zero on constant columns
        ranked_results['reasons'] = [
            explain_match(seed_vector, feature_matrix[idx], feature_cols, feature_ranges)
            for idx in top_indices
        ]
        columns = columns + ['reasons']

    return ranked_results[columns]