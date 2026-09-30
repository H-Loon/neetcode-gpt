import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)

        # Collect all words to build the vocabulary
        all_words = set()
        for sentence in positive + negative:
            all_words.update(sentence.split())
        
        # Sort and map to integers starting at 1
        vocab = {word: i + 1 for i, word in enumerate(sorted(all_words))}
        
        # Encode sentences
        all_tensors = []
        for sentence in positive + negative:
            encoded = [vocab[word] for word in sentence.split()]
            all_tensors.append(torch.tensor(encoded))
            
        # Pad sequences
        # pad_sequence handles the variable lengths automatically
        result = torch.nn.utils.rnn.pad_sequence(all_tensors, batch_first=True, padding_value=0)
        
        return result.float()


