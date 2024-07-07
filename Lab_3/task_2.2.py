import pandas as pd
from scipy.sparse.linalg import svds
import numpy as np

ratings_file_path = 'ratings.csv'
movies_file_path = 'movies.csv'
df = pd.read_csv(ratings_file_path)
movies = pd.read_csv(movies_file_path)

# створення матриці оцінок
ratings_matrix = df.pivot(index='userId', columns='movieId', values='rating')

# прибирання частини даних
ratings_matrix = ratings_matrix.dropna(thresh=30, axis=0)
ratings_matrix = ratings_matrix.dropna(thresh=30, axis=1)

# заміна NaN на середнє значення у кожному стовпці
filled_ratings_matrix_mean = ratings_matrix.apply(lambda col: col.fillna(col.mean()), axis=0)

# перетворення DataFrame в NumPy array
R = filled_ratings_matrix_mean.values

# нормалізувати оцінки - відняти від кожного рядка середню оцінку, яку давав цей користувач
user_ratings_mean = np.mean(R, axis=1)  # середнє рейтингів кожного юзера
R_demeaned = R - user_ratings_mean.reshape(-1, 1)

# SVD
U, sigma, Vt = svds(R_demeaned, k=3)  # k відповідає за розмірність даних, яку зберігаємо -- ознака
# U розмірністю n_users * 3

sigma_diagonal = np.diag(sigma)

# матриця з прогнозованими оцінками
all_user_predicted_ratings = np.dot(np.dot(U, sigma_diagonal), Vt) + user_ratings_mean.reshape(-1, 1)
preds_df = pd.DataFrame(all_user_predicted_ratings, columns=ratings_matrix.columns, index=ratings_matrix.index)

# копія початкової матриці оцінок
original_ratings_matrix = ratings_matrix.copy()

# заміна існуючих оцінок на NaN у новій таблиці
predicted_only_ratings_matrix = preds_df.mask(original_ratings_matrix.notna())


def get_top_recommendations(user_id, num_recommendations=10):
    if user_id not in predicted_only_ratings_matrix.index:
        print(f"User ID {user_id} not found in the ratings matrix.")
        return None

    user_predictions = predicted_only_ratings_matrix.loc[user_id]

    # прогноз оцінок
    sorted_user_predictions = user_predictions.sort_values(ascending=False)

    # рекомендовані 10 фільмів
    top_indices = sorted_user_predictions.index[:num_recommendations]

    movie_titles = movies.loc[top_indices, 'title'].tolist()
    movie_genres = movies.loc[top_indices, 'genres'].tolist()

    recommendations_table = pd.DataFrame({'Title': movie_titles, 'Genres': movie_genres})

    return recommendations_table


recommendations = get_top_recommendations(1)
print(recommendations)
print()

recommendations = get_top_recommendations(2)
print(recommendations)
print()

recommendations = get_top_recommendations(3)
print(recommendations)
print()