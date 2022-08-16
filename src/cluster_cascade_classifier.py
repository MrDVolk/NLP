import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier


class ClusterCascadeClassifier:
    def __init__(self, *, clustering_func, cluster_classifier_factory, label_classifier_factory, vectorize_func=None):
        self.clustering_func = clustering_func
        self.cluster_classifier_factory = cluster_classifier_factory
        self.label_classifier_factory = label_classifier_factory

        if vectorize_func is None:
            self.vectorize_func = lambda x: x
        self.vectorize_func = vectorize_func

        self.cluster_classifier = None
        self.cluster_classifiers_map = dict()

    def fit(self, entries, labels):
        frame = pd.DataFrame(data={'data': entries, 'label': labels})
        frame['vector'] = frame['data'].apply(self.vectorize_func)
        frame['cluster'] = self.clustering_func(frame)

        cluster_vectors, clusters = frame['data'].reset_index(drop=True), frame['cluster'].to_numpy()
        self.cluster_classifier = self.cluster_classifier_factory().fit(cluster_vectors, clusters)

        for cluster, cluster_df in frame.groupby(by='cluster'):
            cluster_x, cluster_y = cluster_df['data'].reset_index(drop=True), cluster_df['label'].to_numpy()

            if len(cluster_df['label'].unique()) == 1:
                self.cluster_classifiers_map[cluster] = DummyClassifier(
                    strategy='constant',
                    constant=cluster_df['label'].to_numpy()[0]
                ).fit(cluster_x, cluster_y)
                continue

            self.cluster_classifiers_map[cluster] = self.label_classifier_factory().fit(cluster_x, cluster_y)

        return self

    def predict(self, entries):
        results = np.zeros((entries.shape[0],))

        cluster_prediction = self.cluster_classifier.predict(entries)
        unique_clusters = np.unique(cluster_prediction)
        for cluster in unique_clusters:
            cluster_idx = cluster_prediction == cluster
            vectors = entries[cluster_idx]
            predictions = self.cluster_classifiers_map[cluster].predict(vectors)
            results[cluster_idx] = predictions

        return results
