import numpy as np
from esprit import esprit
from joint_diag import joint_diag
from numpy.linalg import pinv

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
    all_Psi = []
    for l in range(M):
        # esprit() expects (snapshots, sensors), so we transpose Z_l.
        Psi_l = esprit(all_Z[l].T, r)
        all_Psi.append(Psi_l)

    # Step 3: Joint Diagonalization of Psi Matrices.
    A = np.hstack(all_Psi)
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