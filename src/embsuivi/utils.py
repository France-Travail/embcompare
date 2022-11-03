def bubble_distance(seq1, seq2):
    total_swap = 0

    n_seq1 = len(seq1)
    n_seq2 = len(seq2)

    order = {e: i for i, e in enumerate(seq2)}
    seq = [e for e in seq1 if e in order]

    for _ in range(len(seq) - 1):
        n_swap = 0

        for j in range(len(seq) - 1):
            if order[seq[j]] > order[seq[j + 1]]:
                seq[j], seq[j + 1] = seq[j + 1], seq[j]

                n_swap += 1

        total_swap += n_swap

        if n_swap == 0:
            break

    return total_swap + max(n_seq2, n_seq1) - len(seq)
