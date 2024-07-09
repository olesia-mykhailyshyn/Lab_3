# A=Um*m(otrhogonal- rotate) * Em*n(diagonal - stretch) V_transponedn*n(otrhogonal - rotate)
import numpy as np

matrix = np.array([[2, 8],
                  [7, 1],
                  [6, 9]])

# def generate_random_matrix(max_size=5):
#     rows = np.random.randint(1, max_size + 1)
#     cols = np.random.randint(1, max_size + 1)
#     return np.random.randint(0, 10, size=(rows, cols))


def svd(matrix):
    Sr = np.dot(matrix, matrix.T)

    eigenvalues, eigenvectors = np.linalg.eig(Sr)
    sorted = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted]
    U = eigenvectors[:, sorted]

    eigenvalues = np.maximum(eigenvalues, 0)
    singular_values = np.sqrt(eigenvalues)

    m, n = matrix.shape
    E = np.zeros((m, n))
    np.fill_diagonal(E, singular_values) #np.fill_diagonal(array, values)

    min_dim = min(m, n)
    V = np.zeros((n, n))
    for i in range(min_dim):
        V[:, i] = np.dot(matrix.T, U[:, i]) / singular_values[i] #ця формула дає одразу нормовані вектори

    return U, E, V.T


#matrix = generate_random_matrix()

print("matrix:")
print(matrix)

U, E, Vt = svd(matrix)

print("\n-------------------U-------------------")
print(U)
print("\n--------------E(sigma)----------------")
print(E)
print("\n-------------V transpose---------------")
print(Vt)
print()

# U_np, sigma_np, Vt_np = np.linalg.svd(matrix)
#
# print("-------------------U (NumPy)-------------------")
# print(U_np)
# print("\n--------------Sigma (NumPy)----------------")
# print(sigma_np)
# print("\n-------------V transpose (NumPy)---------------")
# print(Vt_np)

print()
started_matrix = np.dot(U, np.dot(E, Vt))
print("Started matrix:")
print(started_matrix)

print("\nOriginal matrix:")
print(matrix)