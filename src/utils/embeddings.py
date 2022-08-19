import mmap
import numpy as np
from tqdm import tqdm
from gensim.models import KeyedVectors


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


def load_embeddings_from_gensim_format(path):
    embeddings = KeyedVectors.load_word2vec_format(path, binary=True)
    embeddings_dict = {}
    for word in tqdm(embeddings.key_to_index):
        embeddings_dict[word] = embeddings[word]

    return embeddings_dict


def average_vectorizations(row, embeddings_dict):
    vectors = []
    for word in row.split():
        if word in embeddings_dict:
            vectors.append(embeddings_dict[word])

    if len(vectors) == 0:
        return np.zeros(300)

    return np.mean(vectors, axis=0)
