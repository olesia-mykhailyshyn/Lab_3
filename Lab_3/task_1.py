# A=Um*m(otrhogonal- rotate) * Em*n(diagonal - stretch) V_transponedn*n(otrhogonal - rotate)
import numpy as np


# def generate_random_matrix(max_size=5):
#     rows = np.random.randint(1, max_size + 1)
#     cols = np.random.randint(1, max_size + 1)
#     return np.random.randint(0, 10, size=(rows, cols))

#matrix = np.array([[5, 3, 2], [9, 7, 2]])
matrix = np.array([[2, 8, 7, 9, 1],
                  [7, 1, 4, 7, 7],
                  [6, 9, 9, 2, 2]])


def is_orthonormal(vectors):
    return np.allclose(np.dot(vectors.T, vectors), np.eye(vectors.shape[1]))


def normalize_vectors(vectors):
    for i in range(vectors.shape[1]):
        vectors[:, i] /= np.linalg.norm(vectors[:, i])
    return vectors


def svd(matrix):
    Sr = np.dot(matrix, matrix.T)

    eigenvalues, eigenvectors = np.linalg.eig(Sr)

    U = eigenvectors

    if not is_orthonormal(U):
        print("Eigenvectors are not orthonormal. Normalizing...")
        U = normalize_vectors(U)

    sorted = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.abs(eigenvalues[sorted])

    #U = eigenvectors[:, sorted]

    singular_values = np.sqrt(eigenvalues)

    E = np.zeros((matrix.shape[0], matrix.shape[1]))
    np.fill_diagonal(E, singular_values)

    V = np.zeros((matrix.shape[1], matrix.shape[1]))
    for i in range(len(singular_values)):
        V[:, i] = np.dot(matrix.T, U[:, i]) / singular_values[i]

    return U, E, V.T


#matrix = generate_random_matrix()

print("Generated random matrix:")
print(matrix)

U, E, Vt = svd(matrix)

print("\n-------------------U-------------------")
print(U)
print("\n--------------E(sigma)----------------")
print(E)
print("\n-------------V transpose---------------")
print(Vt)
print()

U_np, sigma_np, Vt_np = np.linalg.svd(matrix)

print("-------------------U (NumPy)-------------------")
print(U_np)
print("\n--------------Sigma (NumPy)----------------")
print(sigma_np)
print("\n-------------V transpose (NumPy)---------------")
print(Vt_np)

print()
reconstructed_matrix = np.dot(U, np.dot(E, Vt))
print("Reconstructed matrix:")
print(reconstructed_matrix)

print("\nOriginal matrix:")
print(matrix)