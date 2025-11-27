import numpy as np
from esprit import esprit
from joint_diag import joint_diag
from numpy.linalg import pinv, eig
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

    # Step 3: Eigenvalue Matching (Brute Force)
    # Since Psi_l matrices are in different bases, we cannot use joint_diag.
    # Instead, we compute eigenvalues for each Psi_l and match them.
    
    # Compute eigenvalues for all l
    # all_evals: (M, r)
    all_evals = np.zeros((M, r), dtype=complex)
    for l in range(M):
        try:
            evals, _ = eig(all_Psi[l])
            all_evals[l] = evals
        except np.linalg.LinAlgError:
            all_evals[l] = np.nan
            
    # Step 4: Reconstruct d-D Frequencies (omega_k).
    # We pick d directions to form a basis and try all permutations.
    # U_directions: (M, d)
    
    # Find d linearly independent directions (usually first d)
    basis_indices = []
    for l in range(M):
        if len(basis_indices) < d:
            basis_indices.append(l)
            # Check independence
            if np.linalg.matrix_rank(U_directions[basis_indices]) < len(basis_indices):
                basis_indices.pop()
                
    if len(basis_indices) < d:
        raise ValueError("Could not find d linearly independent directions.")
        
    # Generate all permutations for the basis directions (except the first one to fix ordering)
    # We fix the order of eigenvalues for the first basis direction.
    # For the other d-1 directions, we try all r! permutations.
    
    import itertools
    perms = list(itertools.permutations(range(r)))
    
    best_omega_hat = None
    min_error = float('inf')
    best_permutation_per_direction = np.zeros((M, r), dtype=int)
    
    # Iterate over all combinations of permutations for the remaining d-1 basis directions
    # num_combinations = (r!)^(d-1)
    # For r=3, d=3 -> 36 combinations.
    
    basis_perms_combinations = list(itertools.product(perms, repeat=d-1))
    
    for combo in basis_perms_combinations:
        # Construct a hypothesis for pairings
        # We collect the "aligned" phases for the basis directions
        # aligned_phases: (d, r)
        aligned_phases = np.zeros((d, r))
        
        # First basis direction: identity permutation
        l0 = basis_indices[0]
        aligned_phases[0] = np.angle(all_evals[l0])
        
        # Other basis directions
        for i, perm_idx in enumerate(combo):
            l = basis_indices[i+1]
            perm = list(perm_idx)
            aligned_phases[i+1] = np.angle(all_evals[l][perm])
            
        # Unwrapping is tricky here because we only have d points.
        # But we assume no wrapping for now (small delta).
        
        # Solve for omega candidates
        # U_basis @ omega = phases/delta
        # omega = inv(U_basis) @ phases/delta
        # U_basis: (d, d)
        # phases: (d, r) -> columns are phi_k
        # omega: (d, r)
        
        U_basis = U_directions[basis_indices]
        omega_candidates = pinv(U_basis) @ (aligned_phases / delta)
        omega_candidates = omega_candidates.T # (r, d)
        
        # Evaluate error on ALL directions
        total_error = 0.0
        
        # For each direction l, find best match for these omega candidates
        current_perms = np.zeros((M, r), dtype=int)
        
        for l in range(M):
            # Predicted phases: delta * u_l^T * omega
            # (r,)
            pred_phases = delta * (U_directions[l] @ omega_candidates.T)
            # Wrap predicted phases to [-pi, pi] for comparison
            pred_phases = np.angle(np.exp(1j * pred_phases))
            
            # Observed phases
            obs_phases = np.angle(all_evals[l])
            
            # Find best permutation to match pred and obs
            # We want to minimize sum |pred - obs|^2 (angular distance)
            
            best_l_error = float('inf')
            best_l_perm = None
            
            for p in perms:
                p = list(p)
                permuted_obs = obs_phases[p]
                # Angular difference
                diff = np.angle(np.exp(1j * (pred_phases - permuted_obs)))
                err = np.sum(diff**2)
                
                if err < best_l_error:
                    best_l_error = err
                    best_l_perm = p
            
            total_error += best_l_error
            current_perms[l] = best_l_perm
            
        if total_error < min_error:
            min_error = total_error
            best_omega_hat = omega_candidates
            best_permutation_per_direction = current_perms.copy()
            
    # Final Refinement
    # Use all directions with the best permutations to re-estimate omega
    
    # Construct aligned unwrapped phases for all M
    # paired_phis: (r, M)
    paired_phis = np.zeros((r, M))
    
    for l in range(M):
        perm = best_permutation_per_direction[l]
        paired_phis[:, l] = np.angle(all_evals[l][perm])
        
    # Unwrap?
    # Since we have random directions, unwrap is not applicable directly.
    # However, if we trust our best_omega_hat, we can "unwrap" relative to it.
    # But for small delta, we assume no wrapping.
    
    unwrapped_phis = paired_phis # No unwrap
    
    # Least squares with all directions
    # omega_hat = 1/delta * (pinv(U) @ phis.T).T
    omega_hat = 1/delta * (pinv(U_directions) @ unwrapped_phis.T).T
    omega_hat = np.abs(np.real(omega_hat))
    
    # Reconstruct a_k needs A_glob which needs omega_hat.
    # But we also need to order the "y_vec" or "A_glob" correctly?
    # a_k estimation uses A_glob which depends on omega_hat.
    # It does NOT depend on Psi permutations directly.
    # So we just need correct omega_hat.
    
    # BUT, wait.
    # a_k is estimated from Z.
    # Z = A_glob @ a_k.
    # A_glob depends on omega_hat.
    # If omega_hat is correct, a_k will be correct.
    # The order of a_k will match order of omega_hat.
    
    # So we are good.



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
