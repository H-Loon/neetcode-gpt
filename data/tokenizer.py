from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed
        tokens = list(corpus)
        merges = []

        # a. count all adjacent pairs ONCE (overlaps included)
        counts = {}
        for a, b in zip(tokens, tokens[1:]):
            counts[(a, b)] = counts.get((a, b), 0) + 1

        for _ in range(num_merges):
            if not counts:
                break

            # b. most frequent, ties -> lexicographically smallest
            best = min(counts, key=lambda p: (-counts[p], p))
            a, b = best
            merged = a + b

            # c. merge non-overlapping, left to right,
            #    and remember which pair positions change
            new_tokens = []
            old_pos = set()  # old pairs (tokens[j], tokens[j+1]) that disappear
            new_pos = set()  # new pairs (new_tokens[k], new_tokens[k+1]) that appear
            n = len(tokens)
            i = 0
            while i < n:
                if i < n - 1 and tokens[i] == a and tokens[i + 1] == b:
                    old_pos.update((i - 1, i, i + 1))
                    new_pos.update((len(new_tokens) - 1, len(new_tokens)))
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            # update counts locally instead of recounting everything
            for j in old_pos:
                if 0 <= j < n - 1:
                    p = (tokens[j], tokens[j + 1])
                    counts[p] -= 1
                    if counts[p] == 0:
                        del counts[p]
            m = len(new_tokens)
            for k in new_pos:
                if 0 <= k < m - 1:
                    p = (new_tokens[k], new_tokens[k + 1])
                    counts[p] = counts.get(p, 0) + 1

            tokens = new_tokens

            # d. record the merge
            merges.append([a, b])

        return merges
