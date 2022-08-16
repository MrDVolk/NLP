import numpy as np
from collections import Counter


def get_frequency_map(dataframe, tokens_col, category_col):
    frequency_map = Counter()
    for entry in dataframe.to_dict('records'):
        text = entry[tokens_col]
        category = entry[category_col]
        for token in text:
            frequency_map[(token, category)] += 1

    frequency_map = dict(frequency_map.items())
    return frequency_map


def get_vector(tokens, categories, frequencies, *, unique=True):
    if unique:
        tokens = set(tokens)

    vector = []
    for category in sorted(categories):
        category_value = 0
        for token in tokens:
            category_value += frequencies.get((token, category), 0)

        vector.append(category_value)

    return np.array(vector)
