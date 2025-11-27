import numpy as np
from esprit import esprit
from joint_diag import joint_diag
from numpy.linalg import pinv
import itertools

def jesprit(all_Z, r, U_directions, p_base_points, delta):
    """
    JESPRIT algorithm for mixed Poisson parameter estimation from PGF samples.

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

    # Step 2: Estimate Rotational Invariance Matrices Psi_l.
    # all_Z is list of M arrays of shape (N, S).
    # We stack them to (M, N, S).
    Z_stack = np.stack(all_Z)
    
    # esprit expects (batch, snapshots, sensors).
    # We treat each direction l as a batch item.
    # Each Z_l has shape (N, S).
    # In the original code, we passed Z_l.T which is (S, N).
    # So snapshots=S, sensors=N.
    # We need input shape (M, S, N).
    Z_input = Z_stack.transpose(0, 2, 1)
    
    # all_Psi: shape (M, r, r)
    all_Psi = esprit(Z_input, r)

    # Step 3: Joint Diagonalization of Psi Matrices.
    # joint_diag expects A of shape (r, M*r).
    # We need to stack the Psi matrices horizontally.
    # all_Psi is (M, r, r).
    # Transpose to (r, M, r) -> reshape to (r, M*r).
    A = all_Psi.transpose(1, 0, 2).reshape(r, M * r)
    T_hat, all_Phi_hat = joint_diag(A)

    # Step 4: Reconstruct d-D Frequencies (omega_k).
    # all_Phi_hat has shape (r, M*r). We reshape to (r, M, r) to access blocks.
    Phi_reshaped = all_Phi_hat.reshape(r, M, r)
    
    # Extract diagonals: paired_phis[k, l] = angle(Phi_reshaped[k, l, k])
    # We use advanced indexing to extract the diagonal elements for each M block.
    diag_elements = Phi_reshaped[np.arange(r), :, np.arange(r)]
    paired_phis = np.angle(diag_elements)

    # Phase unwrap along each row.
    unwrapped_phis = np.unwrap(paired_phis, axis=1)

    # Step 3: Least squares to estimate omegas
    omega_hat = 1/delta*(pinv(U_directions) @ unwrapped_phis.T).T
    omega_hat = np.abs(np.real(omega_hat))

    # Step 5: Amplitude Estimation (pi_k).    
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
