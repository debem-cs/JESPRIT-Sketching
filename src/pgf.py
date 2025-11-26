import numpy as np

def PGF(X, t):
    """
    Computes the empirical Probability Generating Function (PGF) from a dataset.
    
    Parameters
    ----------
    X : np.ndarray
        The data matrix of shape (n_samples, d_counts).
    t : np.ndarray
        The vector of variables of shape (d_counts).
        
    Returns
    -------
    complex
        The estimated PGF value.
    """
    if X.ndim != 2:
        raise ValueError("Input data X must be a 2D array.")
    if t.ndim != 1:
        raise ValueError("Input t must be a 1D array.")
    if X.shape[1] != t.shape[0]:
        raise ValueError("The number of features in X and the length of t must be the same.")

    log_t = np.log(t)
    dot_product = X @ log_t
    exp_values = np.exp(dot_product)

    return np.mean(exp_values)

def sample_PGF(X, M, S, N, delta):
    """
    Generates measurement matrices by sampling the PGF along lines.
    
    Parameters
    ----------
    X : np.ndarray
        The data matrix of shape (n_samples, d).
    M : int
        Number of directions.
    S : int
        Number of snapshots (base points) per direction.
    N : int
        Number of samples along each line.
    delta : float
        Scaling factor for the grid to keep |t| close to 1.
        
    Returns
    -------
    all_Z : list of np.ndarray
        A list containing M measurement matrices Z_l, each of shape (N, S).
    U_directions : np.ndarray
        The randomly generated direction vectors, shape (M, d).
    p_base_points : np.ndarray
        The randomly generated base points, shape (M, S, d).
    """

    d = X.shape[1]

    # Generate random direction vectors
    U_directions = np.random.randn(M, d)
    
    # Normalize each direction vector to have a unit norm.
    norms = np.linalg.norm(U_directions, axis=1, keepdims=True)
    U_directions = np.where(norms > 1e-10, U_directions / norms, U_directions)

    # Generate base points
    p_base_points = np.random.randn(M, S, d)

    # Pre-compute n vector for broadcasting
    # shape (1, N, 1, 1)
    n_vec = np.arange(N).reshape(1, N, 1, 1)
    
    # Reshape U_directions for broadcasting: (M, 1, 1, d)
    U_reshaped = U_directions.reshape(M, 1, 1, d)
    
    # Reshape p_base_points for broadcasting: (M, 1, S, d)
    P_reshaped = p_base_points.reshape(M, 1, S, d)
    
    # u: shape (M, N, S, d)
    # Broadcasting: (1, N, 1, 1) * (M, 1, 1, d) + (M, 1, S, d) -> (M, N, S, d)
    u = n_vec * U_reshaped + P_reshaped
    
    # t: shape (M, N, S, d)
    t = 1.0 + 1j * delta * u
    
    # log_t: shape (M, N, S, d)
    log_t = np.log(t)
    
    # Transpose X for matrix multiplication: (d, n_samples)
    X_T = X.T
    
    # Compute dot product with data
    # log_t is (M, N, S, d), X_T is (d, n_samples)
    # Result is (M, N, S, n_samples)
    dot_product = log_t @ X_T
    
    # Exponentiate
    exp_values = np.exp(dot_product)
    
    # Mean over samples (axis 3)
    # Z_all_array: shape (M, N, S)
    Z_all_array = np.mean(exp_values, axis=3)
    
    # Convert to list of (N, S) arrays
    all_Z = [Z_all_array[l] for l in range(M)]
        
    return all_Z, U_directions, p_base_points
