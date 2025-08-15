# https://github.com/nicklashansen/rnn_lstm_from_scratch/blob/master/RNN_LSTM_from_scratch.ipynb

import numpy as np
from collections import defaultdict

np.random.seed(42)

def generate_dataset(num_sequences=2**8):
    samples = []
    for _ in range(num_sequences):
        num_tokens = np.random.randint(1, 12)
        sample = ['a'] * num_tokens + ['b'] * num_tokens + ['EOS']
        samples.append(sample)
    return samples

sequences = generate_dataset()

print(sequences[0])

# word_to_idx = {'a': 0, 'b': 1, 'c': 2}
# idx_to_word = {v: k for k, v in word_to_idx}

def sequences_to_dict(sequences):
    flatten = lambda l: [item for sublist in l for item in sublist]
    all_words = flatten(sequences)

    word_count = defaultdict(int)
    for word in all_words:
        word_count[word] += 1

    word_count
