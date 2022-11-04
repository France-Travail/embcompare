from embsuivi import sequences_similarity


def test_damerau_levenshtein_distance():

    distance = sequences_similarity.damerau_levenshtein_distance

    assert distance("abcd", "abcd") == 0
    assert distance("abcd", "acbd") == 1
    assert distance("abcd", "wxyz") == 4
    assert distance("abcde", "zabcd") == 2

    assert distance(["a", "b", "c", "d"], ["a", "c", "b", "d"]) == 1


def test_damerau_levenshtein_similarity():
    similarity = sequences_similarity.damerau_levenshtein_similarity

    assert similarity("abcd", "abcd") == 1
    assert similarity("abcd", "wxyz") == 0
    assert similarity("abcd", "acbd") == 1 - 1 / 4
