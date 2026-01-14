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

    d = X.shape[1]

    # Generate random direction vectors
    U_directions = np.random.randn(M, d)
    
    # Normalize each direction vector to have a unit norm.
    norms = np.linalg.norm(U_directions, axis=1, keepdims=True)
    U_directions = np.where(norms > 1e-10, U_directions / norms, U_directions)

    # Generate base points
    # IMPORTANT: For Global SVD / Joint ESPRIT, the base points (snapshots) must be
    # shared across all directions to ensure the source signals are coherent.
    # So we generate one set of base points and reuse it.
    p_shared = np.random.randn(S, d)
    # Broadcast to (M, S, d)
    p_base_points = np.tile(p_shared, (M, 1, 1))

    # Pre-compute n vector for broadcasting
    # shape (1, N, 1, 1)
    n_vec = np.arange(N).reshape(1, N, 1) # (N, 1) for broadcasting against (S, d)? No.
    
    # We will compute Z for each direction l separately to save memory
    all_Z = []
    
    # Transpose X once for efficiency: (d, n_samples)
    X_T = X.T
    
    for l in range(M):
        # direction u_l: (d,)
        u_l = U_directions[l] # (d,)
        
        # base points p_l: (S, d)
        # p_l is the same for all l now (conceptually), but we take the slice
        p_l = p_base_points[l] # (S, d)
        
        # We need to compute points P_lns = p_l_s + n * u_l
        # target shape: (N, S, d)
        
        # Reshape u_l: (1, 1, d)
        u_l_resh = u_l.reshape(1, 1, d)
        
        # Reshape p_l: (1, S, d)
        p_l_resh = p_l.reshape(1, S, d)
        
        # n_vec reshaped: (N, 1, 1)
        n_vec_resh = np.arange(N).reshape(N, 1, 1)
        
        # Compute u coordinates: (N, S, d)
        u_coords = n_vec_resh * u_l_resh + p_l_resh
        
        # t = 1 + j * delta * u_coords
        t = 1.0 + 1j * delta * u_coords
        
        # log_t: (N, S, d)
        log_t = np.log(t)
        
        # Compute dot product with data X_T (d, n_samples)
        # log_t is (N, S, d)
        # Result is (N, S, n_samples)
        dot_product = log_t @ X_T
        
        # Exponentiate
        exp_values = np.exp(dot_product)
        
        # Mean over samples (axis 2) -> (N, S)
        Z_l = np.mean(exp_values, axis=2)
        
        all_Z.append(Z_l)
        
    return all_Z, U_directions, p_base_points
