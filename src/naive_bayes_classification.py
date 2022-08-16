from collections import Counter
import numpy as np


class NaiveBayesClassifier:
    def __init__(self):
        self.unique_labels = None
        self.vocabulary = set()
        self.frequency_map = dict()
        self.logprior_map = dict()
        self.loglikelihood = dict()
        self.trained = False

    def fit(self, entries, labels):
        self.unique_labels = sorted(np.unique(labels))
        self.create_frequency_map(entries, labels)
        self.create_loglikelihood(labels)
        self.trained = True
        return self

    def predict(self, entries):
        results = []
        for entry in entries:
            labels_vector = []
            for label in self.unique_labels:
                label_likelihood = self.logprior_map[label]
                for word in entry:
                    label_likelihood += self.loglikelihood.get((word, label), 0)

                labels_vector.append(label_likelihood)

            softmaxed_likelihoods = self.softmax(labels_vector)
            results.append(self.unique_labels[np.argmax(softmaxed_likelihoods)])

        return results

    def create_frequency_map(self, entries, labels):
        frequency_map = Counter()
        for i, category in enumerate(labels):
            text = entries[i]
            for token in text:
                frequency_map[(token, category)] += 1

        self.frequency_map = dict(frequency_map.items())

    def create_loglikelihood(self, train_y):
        self.vocabulary = {word for word, _ in self.frequency_map}
        vocabulary_length = len(self.vocabulary)

        word_count = self.get_number_of_words(filter_func=lambda _: True)
        document_count = len(train_y)

        for label in self.unique_labels:
            this_label_word_count = self.get_number_of_words(filter_func=lambda y: y == label)
            other_labels_word_count = word_count - this_label_word_count

            this_label_document_count = len([y for y in train_y if y == label])
            other_labels_document_count = document_count - this_label_document_count

            label_logprior = np.log(this_label_document_count) - np.log(other_labels_document_count)
            self.logprior_map[label] = label_logprior

            for word in self.vocabulary:
                this_label_frequency = self.frequency_map.get((word, label), 0)
                other_label_frequency = sum([
                    self.frequency_map.get((word, other_label), 0)
                    for other_label in self.unique_labels
                    if other_label != label
                ])

                this_label_word_probability = (this_label_frequency + 1) / (this_label_word_count + vocabulary_length)
                other_label_word_probability = (other_label_frequency + 1) / (other_labels_word_count + vocabulary_length)

                self.loglikelihood[(word, label)] = np.log(this_label_word_probability / other_label_word_probability)

    def inspect_likelihoods(self, target_label, *, n=100, min_other_appearances=0):
        if not self.trained:
            return None

        results = []
        for word in self.vocabulary:
            target_appearances = self.frequency_map.get((word, target_label), 0)
            other_appearances = sum([
                self.frequency_map.get((word, other_label), 0)
                for other_label in self.unique_labels
                if other_label != target_label
            ])
            if other_appearances < min_other_appearances:
                continue

            results.append({
                'word': word,
                'target_appearances': target_appearances,
                'other_appearances': other_appearances,
                'ratio': round((target_appearances + 1) / (other_appearances + 1), 4),
                'loglikelihood': self.loglikelihood.get((word, target_label), 0)
            })

        results = sorted(results, key=lambda x: x['ratio'], reverse=True)
        return results[:n]

    def inspect_word(self, word):
        if not self.trained:
            return None

        results = {'word': word}
        for label in self.unique_labels:
            appearances = self.frequency_map.get((word, label), 0)
            results[f'appearances for {label}'] = appearances

        return results

    def get_number_of_words(self, *, filter_func):
        return sum([
            self.frequency_map[(word, label)]
            for word, label in self.frequency_map
            if filter_func(label)
        ])

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
