from typing import Hashable, Sequence


def damerau_levenshtein_distance(s1: Sequence[Hashable], s2: Sequence[Hashable]) -> int:
    """Unrestricted Damerau-Levenshtein distance between two sequences

    Implementation from https://github.com/life4/textdistance

    Compute edit distance according to following operations :
        * deletion:      ABC -> BC, AC, AB
        * insertion:     ABC -> ABCD, EABC, AEBC..
        * substitution:  ABC -> ABE, ADC, FBC..
        * transposition: ABC -> ACB, BAC

    Args:
        s1 (Sequence[Hashable]): first sequence
        s2 (Sequence[Hashable]): second sequence

    Returns:
        int: returns the damerau-levensthein distance between the two sequences

    https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance
    """
    d = {}
    da = {}

    len1 = len(s1)
    len2 = len(s2)

    maxdist = len1 + len2
    d[-1, -1] = maxdist

    # distance matrix initialization
    for i in range(len(s1) + 1):
        d[i, -1] = maxdist
        d[i, 0] = i

    for j in range(len(s2) + 1):
        d[-1, j] = maxdist
        d[0, j] = j

    for i, cs1 in enumerate(s1, start=1):
        db = 0
        for j, cs2 in enumerate(s2, start=1):
            i1 = da.get(cs2, 0)
            j1 = db
            if cs1 == cs2:
                cost = 0
                db = j
            else:
                cost = 1

            d[i, j] = min(
                d[i - 1, j - 1] + cost,  # substitution
                d[i, j - 1] + 1,  # insertion
                d[i - 1, j] + 1,  # deletion
                d[i1 - 1, j1 - 1] + (i - i1) - 1 + (j - j1),  # transposition
            )
        da[cs1] = i

    return d[len1, len2]


def damerau_levenshtein_similarity(
    s1: Sequence[Hashable], s2: Sequence[Hashable]
) -> float:
    """Similarity based on unrestricted Damerau-Levenshtein distance between two
    sequences

    The similarity is computed as 1 - DL-distance(s1, s2) / max(len(s1), len(s2))

    Args:
        s1 (Sequence[Hashable]): first sequence
        s2 (Sequence[Hashable]): second sequence

    Returns:
        float: returns the damerau-levensthein similarity between the two sequences
    """
    return 1 - damerau_levenshtein_distance(s1, s2) / max(len(s1), len(s2))
