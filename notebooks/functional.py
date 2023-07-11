import numpy as np


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


def rff_gaussian(X, n_components, gamma):
    n_features = X.shape[1]
    W = np.random.normal(0, np.sqrt(2 * gamma), (n_features, n_components))
    b = np.random.uniform(0, 2 * np.pi, n_components)
    Z = np.sqrt(2.0 / n_components) * np.cos(X.dot(W) + b)
    return Z


def rffkmsm(data, n_dims):
    # Assume 'data' is your original matrix with samples in rows and features in columns

    # Standardize the data
    standardized_data = data

    # Map the data using RFF
    # Set the gamma parameter for the Gaussian kernel
    gamma = 1 / data.shape[1]
    n_components = n_dims  # Set the number of random features for RFF
    Z = rff_gaussian(standardized_data, n_components, gamma)

    # Calculate the approximate kernel matrix
    approx_kernel_matrix = Z.dot(Z.T)

    # Center the approximate kernel matrix
    row_mean = np.mean(approx_kernel_matrix, axis=0)
    col_mean = np.mean(approx_kernel_matrix, axis=1)
    matrix_mean = np.mean(approx_kernel_matrix)
    centered_approx_kernel_matrix = approx_kernel_matrix - \
        row_mean - col_mean[:, np.newaxis] + matrix_mean

    # Compute the eigenvalues and eigenvectors of the centered approximate kernel matrix
    _, V = np.linalg.eigh(centered_approx_kernel_matrix)
    # Step 5: Select the top k eigenvectors
    return V[:, -n_dims:]


def gaussian_kernel(x, y, gamma):
    return np.exp(-gamma * np.linalg.norm(x - y)**2)


def standardize(data):
    mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    return (data - mean) / std_dev


def kmsm(data, n_dims):
    # Standardize the data
    standardized_data = standardize(data)

    # Step 1: Calculate the kernel matrix
    n_samples = standardized_data.shape[0]
    kernel_matrix = np.zeros((n_samples, n_samples))
    gamma = 1 / standardized_data.shape[1]  # Consider tuning this parameter

    for i in range(n_samples):
        for j in range(n_samples):
            kernel_matrix[i, j] = gaussian_kernel(
                standardized_data[i], standardized_data[j], gamma)

    # Step 2: Center the kernel matrix
    row_mean = np.mean(kernel_matrix, axis=0)
    col_mean = np.mean(kernel_matrix, axis=1)
    matrix_mean = np.mean(kernel_matrix)
    centered_kernel_matrix = kernel_matrix - \
        row_mean - col_mean[:, np.newaxis] + matrix_mean

    # Step 3: Compute the eigenvalues and eigenvectors of the centered kernel matrix
    _, V = np.linalg.eigh(centered_kernel_matrix)

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
