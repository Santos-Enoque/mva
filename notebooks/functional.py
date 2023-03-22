import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import KernelPCA
from sklearn.metrics.pairwise import pairwise_kernels

def pca(data, n_dims):
    data = data.T
# Step 1: Standardize the data
    mean = np.mean(data, axis=0)
    standardized_data = (data - mean) 
# Step 2: Calculate the covariance matrix
    cov_matrix = np.cov(standardized_data.T)

# Step 3: Compute the eigenvalues and eigenvectors of the covariance matrix
    _, V = np.linalg.eigh(cov_matrix)


# Step 5: Select the top k eigenvectors
    return V[:, -n_dims:]

def cosine_similarities(X, subspace_bases):
    """
    Computes cosine similarities of all input data to all input subspaces.
    Args:
        X (2d_array):   2d-array of data where columns are data samples
        subspace_bases (3d-array):  an array of all class subspaces
    Returns:
        A 2d-array where rows represent similarities to class subspaces.
    """
    proj = np.dot(subspace_bases.transpose(0, 2, 1), X)
    norm = np.linalg.norm(X, axis=0)
    sim = ((proj ** 2) / (norm ** 2)).sum(axis=1)

    return sim.T


def canonical_similarity(ref_subspaces, input_subspace):
    """
    Computes similarity between reference subspaces and an input subspace based on
    canonical angles between subspaces.
    Args:
        ref_subspaces (3d-array): A 3d-array of reference subspaces of shape
            (NUM_OF_CLASSES x NUM_OF_FEATURES x NUM_OF_DIMS).
        input_subspace (2d-array): A 2d-array representing basis vectors of input subspace.
    Returns:
        A list of similarities (cosines) between 0 and 1 based on canonical angles.
    """
    gramians = ref_subspaces.transpose(0, 2, 1) @ input_subspace
    cosines = [canonical_angles(g) for g in gramians]

    return cosines


def canonical_angles(gramian):
    """
    Computes cosines of canonical angles based on a X.T @ Y gramian matrix, where
    X is a tensor of all refererence subspaces and Y is an input subspace.
    Args:
        gramian (3d-array): A 3d-array of gramians shaped (NUM_OF_CLASSES x
            NUM_OF_DIMS x NUM_OF_DIMS), where NUM_OF_DIMS is the number of subspace
            dimensions.
    Returns:
        A list of cosines of canonical angles.
    """
    cosines = (gramian * gramian).sum() / min(gramian.shape)
    # cosines = np.linalg.norm(g)  # Frobenius norm, more straight-forward way to implement

    return cosines



