import numpy as np


def pca(data: np.array, l: int = 2) -> np.array:
    """
    Perform PCA transformation on data

    :param data: data you want to transform
    :param l: desired resulting dimensions
    :returns: transformed data
    """
    A = data.T @ data
    eigen_values, eigen_vectors = np.linalg.eig(A)
    indices = eigen_values.argsort()[-l:][::-1]
    D = np.array(eigen_vectors[indices])

    return data @ D.T
