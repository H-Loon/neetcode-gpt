from typing import List, Dict



class Solution:
    def _tokenize(self, s: str, vocab: Dict[str, int]) -> List[str]:
        # Greedy left-to-right longest match; no match -> consume 1 character
        max_len = max((len(k) for k in vocab), default=1)
        tokens = []
        i = 0
        n = len(s)
        while i < n:
            for j in range(min(n, i + max_len), i, -1):
                if s[i:j] in vocab:
                    tokens.append(s[i:j])
                    i = j
                    break
            else:
                tokens.append(s[i])
                i += 1
        return tokens

    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # Tokenize each number using greedy left-to-right longest match.
        # Return a list of token lists showing how each number gets split.
        return [self._tokenize(str(num), vocab) for num in numbers]

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Count how many tokens the text uses with greedy tokenization.
        # Use greedy left-to-right longest match.
        return len(self._tokenize(text, vocab))

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.
        words = text.split()
        if not words:
            return 0.0
        return round(self.count_tokens(text, vocab) / len(words), 4)
