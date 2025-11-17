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
    paired_phis = np.zeros((r, M))
    for l in range(M):
        # Extract the l-th diagonalized matrix.
        Phi_l = all_Phi_hat[:, l*r:(l+1)*r]
        # The diagonal contains the eigenvalues, whose angles are the 1D frequencies.
        diag_elements = np.diag(Phi_l)
        paired_phis[:, l] = 1/delta*np.angle(diag_elements)

    # Phase unwrap along each row.
    unwrapped_phis = np.unwrap(paired_phis, axis=1)

    # Step 3: Least squares to estimate omegas
    omega_hat = (pinv(U_directions) @ unwrapped_phis.T).T
    omega_hat = np.abs(np.real(omega_hat))

    # Step 5: Amplitude Estimation (pi_k).    
    num_total_samples = M * S * N
    A_glob = np.zeros((num_total_samples, r), dtype=np.complex128)
    y_vec = np.zeros(num_total_samples, dtype=np.complex128)
    
    i = 0 # Global sample index
    for l in range(M):
        for s in range(S):
            for n in range(N):
                n_vec = p_base_points[l, s, :] + n * U_directions[l, :]
                A_glob[i, :] = np.exp(1j * np.dot(omega_hat, delta*n_vec))
                y_vec[i] = all_Z[l][n, s]
                i += 1

    a_k = pinv(A_glob) @ y_vec
    a_k = np.abs(a_k)
    a_k = a_k / np.sum(a_k)
    
    return omega_hat, a_k