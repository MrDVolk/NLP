import numpy as np
import pandas as pd
import warnings
from sklearn.cluster import OPTICS, KMeans
from sklearn.preprocessing import MinMaxScaler


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_distance_frame(dataframe, group_column, vector_column='vector'):
    """Display with distance_frame.style.background_gradient(cmap='coolwarm')"""

    results = []
    for i, first_row in dataframe.iterrows():
        result_row = {'index': first_row[group_column]}
        for j, second_row in dataframe.iterrows():
            distance = 1 - round(cosine_similarity(first_row[vector_column], second_row[vector_column]), 4)
            result_row[second_row[group_column]] = distance

        results.append(result_row)

    return pd.DataFrame(results).set_index('index')


def get_vectorized_group_description(dataframe, group_column='label', vector_column='vector'):
    group_vectors = []

    for group_value, group_df in dataframe.groupby(by=group_column):
        mean_vector = np.mean(np.vstack(group_df[vector_column]), axis=0)
        group_vectors.append({group_column: group_value, vector_column: mean_vector})

    return pd.DataFrame(group_vectors)


def clusterize_by_distance(dataframe, group_column='label', vector_column='vector'):
    group_description_df = get_vectorized_group_description(dataframe, group_column, vector_column)
    distance_matrix = get_distance_frame(group_description_df, group_column, vector_column).to_numpy()
    scaled_matrix = MinMaxScaler(feature_range=(0, 1)).fit_transform(distance_matrix.reshape((-1, 1)))
    scaled_matrix = scaled_matrix.reshape(distance_matrix.shape)

    optics = OPTICS(metric='precomputed', min_samples=2)
    group_description_df['cluster'] = optics.fit_predict(scaled_matrix)

    return map_group_to_cluster(dataframe, group_column, group_description_df)


def clusterize_by_vectors(dataframe, group_column='label', vector_column='vector', *, cluster_count=3, random_state=None):
    group_description_df = get_vectorized_group_description(dataframe, group_column, vector_column)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kmeans = KMeans(n_clusters=cluster_count, random_state=random_state)
        group_description_df['cluster'] = kmeans.fit_predict(np.vstack(group_description_df[vector_column]))

    return map_group_to_cluster(dataframe, group_column, group_description_df)


def map_group_to_cluster(dataframe, group_column, group_description_df):
    group_to_cluster_map = {}
    for row in group_description_df.to_dict('records'):
        group_to_cluster_map[row[group_column]] = row['cluster']
    return dataframe[group_column].apply(lambda group: group_to_cluster_map[group])
