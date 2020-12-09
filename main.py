import os
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

data_path = "./data_set"
movies_name = data_path + "/movies.csv"
movies_rating = data_path + "/ratings.csv"

data_movies = pd.read_csv(movies_name, usecols=['movieId', 'title'], dtype={
    'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})

data_rating = pd.read_csv(movies_rating, usecols=['userId', 'movieId', 'rating'], dtype={
                          'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})

matrix = data_rating.pivot(
    index='movieId', columns='userId', values='rating').fillna(0)

matrix_data = csr_matrix(matrix.values)
model_knn = NearestNeighbors(
    metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
model_knn.fit(matrix)
hashmap = {
    movie: i for i, movie in
    enumerate(list(data_movies.set_index(
        'movieId').loc[matrix.index].title))
}


def get_recommendation(title):
    index = hashmap[title]
    distances, indices = model_knn.kneighbors(
        matrix.iloc[index].values.reshape(1, -1), n_neighbors=10)
    raw_recommends = sorted(list(zip(indices.squeeze().tolist(
    ), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]
    reverse_mapper = {v: k for k, v in hashmap.items()}
    for i, (idx, dist) in enumerate(raw_recommends):
        print('{0}: {1}, with distance of {2}'.format(
            i+1, reverse_mapper[idx], dist))
    return index


get_recommendation("Inception (2010)")
