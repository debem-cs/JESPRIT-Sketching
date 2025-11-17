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
    for i in range(M):
        norm = np.linalg.norm(U_directions[i,:])
        if norm > 1e-10:
            U_directions[i,:] /= norm

    # Generate base points
    p_base_points = np.random.randn(M, S, d)

    all_Z = []
    for l in range(M):
        # Z_l has shape (N, S)
        Z_l = np.zeros((N, S), dtype=np.complex128)
        for s in range(S):
            for n in range(N):
                # Define the point 'u' in the complex domain
                # We scale the step 'n' by 'scale' (equivalent to Delta in the paper)
                u = p_base_points[l, s, :] + n * U_directions[l, :]
                
                # Map 'u' to the PGF argument 't'
                # t = 1 + i*u
                t = np.ones(d) + 1j * delta*u
                
                # Sample the PGF at this point
                Z_l[n, s] = PGF(X, t)
        all_Z.append(Z_l)
        
    return all_Z, U_directions, p_base_points
