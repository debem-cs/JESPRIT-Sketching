import numpy as np
from joint_diag import joint_diag
from numpy.linalg import pinv, eig, svd
import itertools
from pgf import sample_PGF

def jesprit(X, r, M, S, N, delta):
    """
    JESPRIT algorithm for mixed Poisson parameter estimation from a dataset X.
    Internally computes PGF samples and uses Global SVD.

    Parameters
    ----------
    X : np.ndarray
        The input count data, shape (n_samples, d).
    r : int
        The number of latent factors (components) in the mixture.
    M : int
        Number of random directions.
    S : int
        Number of base points per direction.
    N : int
        Number of evaluation points per line.
    delta : float
        Scaling factor for the grid.

    Returns
    -------
    omega_hat : np.ndarray
        The estimated poisson rates (parameters of the Poisson distributions),
        of shape (r, d).
    a_k : np.ndarray
        The estimated mixture weights of the components, of shape (r,).
    """
    # Step 1: Compute PGF samples
    all_Z, U_directions, p_base_points = sample_PGF(X, M, S, N, delta)
    
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
    
    # omega_hat shape: (d, r)
    # pinv(U): (d, M)
    # phis: (M, r)
    omega_hat = 1/delta * (pinv(U_directions) @ phis)
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
    # omega_hat: (d, r)
    # exponent: (M*S*N, r)
    exponent = 1j * delta * (N_VEC_flat @ omega_hat)
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

from scipy.optimize import linear_sum_assignment

def compute_error(lambda_true, pi_true, lambda_est, pi_est):
    """
    Computes the estimation error, handling permutation ambiguity using the Hungarian Algorithm.
    Returns (rate_error, weight_error, lambda_aligned, pi_aligned).
    """
    r = len(pi_true)
    
    # Cost matrix C[i, j] = error if true_component_i matches est_component_j
    # We define error as Mean Relative Error (MRE) for rates + MRE for weights
    
    # lambda_true: (d, r)
    # lambda_est: (d, r)
    
    # Validated Safe Division function for broadcasting
    def safe_rel_diff(true_vals, est_vals):
        # true_vals: (..., r, 1) or equivalent broadcast-ready shape
        # est_vals:  (..., 1, r)
        diff = np.abs(true_vals - est_vals)
        denom = np.abs(true_vals)
        return np.divide(diff, denom, out=np.zeros_like(diff), where=denom!=0)

    # Rate Cost (r, r)
    # lambda_true: (d, r) -> (d, r, 1)
    lambda_true_exp = lambda_true[:, :, np.newaxis]
    # lambda_est: (d, r) -> (d, 1, r)
    lambda_est_exp = lambda_est[:, np.newaxis, :]
    
    # rel_diff_rate: (d, r, r)
    rel_diff_rate = safe_rel_diff(lambda_true_exp, lambda_est_exp)
    # Mean over dimension d -> (r, r)
    C_rate = np.mean(rel_diff_rate, axis=0)

    # Weight Cost (r, r)
    # pi_true: (r,) -> (r, 1)
    pi_true_exp = pi_true[:, np.newaxis]
    # pi_est: (r,) -> (1, r)
    pi_est_exp = pi_est[np.newaxis, :]
    
    C_weight = safe_rel_diff(pi_true_exp, pi_est_exp)
    
    C = C_rate + C_weight
            
    # Solve assignment problem
    # row_ind will be 0..r-1 (true components)
    # col_ind will be the matching indices in estimated components
    row_ind, col_ind = linear_sum_assignment(C)
    
    # Reorder estimated components to match true components
    # col_ind[i] tells us which estimated component j matches true component i
    best_perm = col_ind
    
    lambda_est_aligned = lambda_est[:, best_perm]
    pi_est_aligned = pi_est[best_perm]
    
    # Recalculate errors with best permutation
    # Rate error (Mean Relative Error over all elements d*r)
    rel_diff_matrix = np.divide(
        np.abs(lambda_true - lambda_est_aligned),
        np.abs(lambda_true),
        out=np.zeros_like(lambda_true),
        where=lambda_true!=0
    )
    rate_err = np.mean(rel_diff_matrix)
    
    # Weight error (Mean Relative Error over r elements)
    rel_diff_weights = np.divide(
        np.abs(pi_true - pi_est_aligned),
        np.abs(pi_true),
        out=np.zeros_like(pi_true),
        where=pi_true!=0
    )
    weight_err = np.mean(rel_diff_weights)
            
    return rate_err, weight_err, lambda_est_aligned, pi_est_aligned
