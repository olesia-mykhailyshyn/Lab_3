import pandas as pd
from scipy.sparse.linalg import svds
import numpy as np
import matplotlib.pyplot as plt

file_path = 'ratings.csv'
df = pd.read_csv(file_path) #зчитування даних і записування їх в таблицю іншого вигляду

ratings_matrix = df.pivot(index='userId', columns='movieId', values='rating') #таблиця в стандартному вигляді

# прибирання частини даних
# user_threshold = 50
# movie_threshold = 50

#юзери по стовпцях, фільми по рядках
# user_ratings_count = ratings_matrix.count(axis=1) #ненульові значення
# filtered_users = user_ratings_count[user_ratings_count >= user_threshold].index
# ratings_matrix = ratings_matrix.loc[filtered_users] #.loc -- доступ
#
# movie_ratings_count = ratings_matrix.count(axis=0)
# filtered_movies = movie_ratings_count[movie_ratings_count >= movie_threshold].index
# ratings_matrix = ratings_matrix.loc[:, filtered_movies]

ratings_matrix = ratings_matrix.dropna(thresh=100, axis=0)
ratings_matrix = ratings_matrix.dropna(thresh=100, axis=1)

# заміна NaN на середнє значення у кожному стовпці
filled_ratings_matrix_mean = ratings_matrix.apply(lambda col: col.fillna(col.mean()), axis=0)
#filled_ratings_matrix_mean = ratings_matrix.fillna(2.5, axis=0)

# перетворення DataFrame в NumPy array
R = filled_ratings_matrix_mean.values

# нормалізувати оцінки - відняти від кожного рядка середню оцінку, яку давав цей користувач
user_ratings_mean = np.mean(R, axis=1) #середнє рейтингів кожного юзера
R_demeaned = R - user_ratings_mean.reshape(-1, 1)

# SVD
U, sigma, Vt = svds(R_demeaned, k=3) #k відповідає за розмірність даних, яку зберігаємо -- ознака
#U розмірністю n_users * 3


def plot_3d_points(matix, labels, title, num_points=20):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(min(num_points, matix.shape[0])):
        ax.scatter(matix[i, 0], matix[i, 1], matix[i, 2])
        ax.text(matix[i, 0], matix[i, 1], matix[i, 2], f'{labels} {i + 1}', size=10)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    plt.show()


plot_3d_points(U, "User", "Users", num_points=10)
plot_3d_points(Vt.T, "Movie", "Films", num_points=10)