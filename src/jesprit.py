import numpy as np
from joint_diag import joint_diag
from numpy.linalg import pinv, eig, svd
import itertools

def jesprit(all_Z, r, U_directions, p_base_points, delta):
    """
    JESPRIT algorithm for mixed Poisson parameter estimation from PGF samples.
    Uses Global SVD to estimate a common signal subspace.

    Parameters
    ----------
    all_Z : list of np.ndarray
        List of M measurement matrices, each of shape (N, S).
    r : int
        The number of latent factors (components) in the mixture.
    U_directions : np.ndarray
        The direction vectors used for sampling, shape (M, d).
    p_base_points : np.ndarray
        The base points used for sampling, shape (M, S, d).
    delta : float
        Scaling factor for the grid to keep |t| close to 1.

    Returns
    -------
    omega_hat : np.ndarray
        The estimated poisson rates (parameters of the Poisson distributions),
        of shape (r, d).
    a_k : np.ndarray
        The estimated mixture weights of the components, of shape (r,).
    """
    # Infer M, N, and S from the dimensions of the input data
    M = len(all_Z)
    if M == 0:
        raise ValueError("all_Z must not be empty.")
    N, S = all_Z[0].shape
    d = U_directions.shape[1]

    # Step 2: Global Subspace Estimation
    # Stack all Z_l along the "sensor" axis (M*N, S)
    X_glob = np.vstack(all_Z)
    
    # SVD of the global data matrix
    # U_glob: (M*N, M*N) if full, but we use full_matrices=False, so (M*N, min(MN, S))
    # We need the first r columns.
    U_glob, _, _ = svd(X_glob, full_matrices=False)
    
    # Global signal subspace of dimension r
    U_s = U_glob[:, :r]

    # Step 3: Build Psi_l from the global subspace
    all_Psi = []
    for l in range(M):
        # Rows corresponding to direction l in the stacked matrix
        start = l * N
        stop = (l + 1) * N
        U_line = U_s[start:stop, :]   # shape (N, r)
        
        # Shifted subspaces along the line
        U_1 = U_line[:-1, :]          # (N-1, r)
        U_2 = U_line[1:, :]           # (N-1, r)
        
        # Rotational invariance matrix for direction l
        Psi_l = pinv(U_1) @ U_2       # (r, r)
        all_Psi.append(Psi_l)

    # Step 4: Joint Diagonalization
    # Stack Psi matrices horizontally: (r, M*r)
    A_joint = np.hstack(all_Psi)
    
    # Solve for the common diagonalizer V
    V, D_blocks = joint_diag(A_joint)
    
    # Extract diagonals (eigenvalues) from the diagonalized blocks
    # mus[l, k] corresponds to direction l, component k
    mus = np.zeros((M, r), dtype=complex)
    for l in range(M):
        D_l = D_blocks[:, l*r:(l+1)*r]
        mus[l, :] = np.diag(D_l)
        
    # Step 5: Parameter Recovery (omega_hat)
    # mus[l, k] = exp(j * delta * u_l^T * omega_k)
    # We recover phases and solve inverse problem
    
    phis = np.angle(mus) # (M, r)
    
    # Assuming small delta, no wrapping handling needed (or very minimal).
    # omega_hat = 1/delta * pinv(U_directions) @ phis
    
    # omega_hat shape: (r, d)
    # pinv(U): (d, M)
    # phis: (M, r)
    omega_hat = 1/delta * (pinv(U_directions) @ phis).T
    omega_hat = np.abs(np.real(omega_hat))

    # Step 6: Amplitude Estimation (a_k / pi_k)
    # We need to construct A_glob using the estimated omega_hat to solve for weights.
    
    # Vectorized construction of A_glob and y_vec
    
    # Reshape for broadcasting to create N_VEC of shape (M, S, N, d)
    # p_base_points: (M, S, d) -> (M, S, 1, d)
    P_broad = p_base_points[:, :, np.newaxis, :]
    # U_directions: (M, d) -> (M, 1, 1, d)
    U_broad = U_directions[:, np.newaxis, np.newaxis, :]
    # n_vals: (N,) -> (1, 1, N, 1)
    n_vals = np.arange(N)[np.newaxis, np.newaxis, :, np.newaxis]
    
    # Compute all points n_vec
    N_VEC = P_broad + n_vals * U_broad
    
    # Flatten to (M*S*N, d) to match the order of loops (l, s, n)
    N_VEC_flat = N_VEC.reshape(-1, d)
    
    # Compute A_glob
    # omega_hat: (r, d)
    # exponent: (M*S*N, r)
    exponent = 1j * delta * (N_VEC_flat @ omega_hat.T)
    A_glob = np.exp(exponent)
    
    # Construct y_vec
    # all_Z is list of (N, S) arrays.
    # We need to flatten in order (l, s, n).
    # Convert to array (M, N, S)
    Z_array = np.array(all_Z)
    # Transpose to (M, S, N) so that flattening corresponds to l, s, n order
    Z_array = Z_array.transpose(0, 2, 1)
    y_vec = Z_array.reshape(-1)

    # Solve linear system: A_glob * a_k = y_vec
    a_k = pinv(A_glob) @ y_vec
    a_k = np.abs(a_k)
    a_k = a_k / np.sum(a_k)
    
    return omega_hat, a_k

def compute_error(lambda_true, pi_true, lambda_est, pi_est):
    """
    Computes the estimation error, handling permutation ambiguity.
    Returns (rate_error, weight_error) for the best permutation.
    """
    r = len(pi_true)
    # Try all permutations to match estimated components to true components
    permutations = list(itertools.permutations(range(r)))
    
    min_total_error = float('inf')
    best_rate_error = 0.0
    best_weight_error = 0.0
    
    for perm in permutations:
        perm = list(perm)
        lambda_est_perm = lambda_est[perm, :].T
        
        # Error in rates (normalized)
        rate_error = np.mean(np.abs(lambda_true - lambda_est_perm))
        
        # Error in weights
        weight_error = np.mean(np.abs(pi_true - pi_est[perm]))
        
        total_error = rate_error + weight_error
        
        if total_error < min_total_error:
            min_total_error = total_error
            best_rate_error = rate_error
            best_weight_error = weight_error
            
    return best_rate_error, best_weight_error
