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
    Optimized to reduce memory usage by looping over directions.
    
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

from joblib import Parallel, delayed

def _compute_single_Z_l(l, u_l, p_l, N, S, d, delta, X):
    # Pre-compute coordinates for this direction
    u_l_resh = u_l.reshape(1, 1, d)
    p_l_resh = p_l.reshape(1, S, d)
    n_vec_resh = np.arange(N).reshape(N, 1, 1) # (N, 1, 1)

    u_coords = n_vec_resh * u_l_resh + p_l_resh
    
    # t = 1 + j * delta * u_coords
    t = 1.0 + 1j * delta * u_coords
    
    # log_t: (N, S, d)
    log_t = np.log(t)
    
    # --- Batch Processing for Memory Efficiency ---
    n_samples = X.shape[0]
    batch_size = 10000 
    Z_l_accum = np.zeros((N, S), dtype=complex)
    
    for i in range(0, n_samples, batch_size):
        end = min(i + batch_size, n_samples)
        X_batch = X[i:end, :] # (batch_size, d)
        
        # Transpose batch for dot product: (d, batch_size)
        X_batch_T = X_batch.T
        
        # log_t: (N, S, d)
        # X_batch_T: (d, batch_size)
        # Result: (N, S, batch_size)
        dot_product = log_t @ X_batch_T
        
        # Exponentiate and sum over batch
        exp_values_batch = np.exp(dot_product)
        Z_l_accum += np.sum(exp_values_batch, axis=2)
        
    # Average over all samples
    Z_l = Z_l_accum / n_samples
    return Z_l

def sample_PGF(X, M, S, N, delta):
    """
    Generates measurement matrices by sampling the PGF along lines.
    Optimized with Batch Processing (Memory) + Parallelization (Speed).
    """
    d = X.shape[1]

    # Generate random direction vectors
    U_directions = np.random.randn(M, d)
    norms = np.linalg.norm(U_directions, axis=1, keepdims=True)
    U_directions = np.where(norms > 1e-10, U_directions / norms, U_directions)

    # Generate base points (shared)
    p_shared = np.random.randn(S, d)
    p_base_points = np.tile(p_shared, (M, 1, 1))
    
    # Parallel execution using threads (shared memory for X)
    # backend="threading" avoids copying X which is crucial for large data
    all_Z = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_compute_single_Z_l)(
            l, 
            U_directions[l], 
            p_base_points[l], 
            N, S, d, delta, X
        ) for l in range(M)
    )
        
    return all_Z, U_directions, p_base_points
