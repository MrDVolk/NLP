import mmap
import numpy as np
from tqdm import tqdm


def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


def load_embeddings(path):
    embeddings_dict = {}
    with open(path, 'r', encoding='utf-8') as file:
        for line in tqdm(file, total=get_num_lines(path)):
            values = line.split()
            word = values[0].lower()
            if word in embeddings_dict:
                continue

            vector = np.asarray(values[1:], dtype='float32')
            embeddings_dict[word] = vector

    return embeddings_dict


def average_vectorizations(row, embeddings_dict):
    vectors = []
    for word in row.split():
        if word in embeddings_dict:
            vectors.append(embeddings_dict[word])

    result_vector = np.mean(vectors, axis=0)
    return result_vector
