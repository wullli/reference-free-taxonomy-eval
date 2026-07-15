from typing import Sequence


def wu_palmer_similarity(path1: Sequence[str], path2: Sequence[str]) -> float:
    """Compute Wu-Palmer similarity for two paths.

    Args:
        path1: Sequence of node ids from root to the first concept.
        path2: Sequence of node ids from root to the second concept.

    Returns:
        Wu-Palmer similarity between the two paths.
    """
    lcs_depth = 0
    for a, b in zip(path1, path2):
        if a == b:
            lcs_depth += 1
        else:
            break

    depth1, depth2 = len(path1), len(path2)
    if depth1 + depth2 == 0:
        return float("nan")
    return (2 * lcs_depth) / (depth1 + depth2)
