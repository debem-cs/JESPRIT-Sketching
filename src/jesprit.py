import numpy as np
from esprit import esprit
from joint_diag import joint_diag
from numpy.linalg import pinv, svd

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

    # Step 2: Global Subspace Estimation (Global SVD)
    # Stack all Z_l along the "sensor" axis (M*N, S)
    X_glob = np.vstack(all_Z)
    
    # SVD of the global data matrix to find common signal subspace
    # U_glob: (M*N, M*N), we need first r columns
    U_glob, _, _ = np.linalg.svd(X_glob, full_matrices=False)
    U_s = U_glob[:, :r]

    # Step 3: Build Rotational Invariance Matrices Psi_l from global subspace
    all_Psi = []
    for l in range(M):
        # Extract rows for direction l
        start = l * N
        stop = (l + 1) * N
        U_line = U_s[start:stop, :]   # (N, r)
        
        # Shifted subspaces
        U_1 = U_line[:-1, :]          # (N-1, r)
        U_2 = U_line[1:, :]           # (N-1, r)
        
        # Rotational invariance matrix
        Psi_l = pinv(U_1) @ U_2       # (r, r)
        all_Psi.append(Psi_l)

    # Step 4: Joint Diagonalization of Psi Matrices.
    # Stack Psi matrices horizontally: (r, M*r)
    A = np.hstack(all_Psi)
    # Solve for common diagonalizer V
    V, D_blocks = joint_diag(A)

    # Step 4: Reconstruct d-D Frequencies (omega_k).
    paired_phis = np.zeros((r, M))
    paired_phis = np.zeros((r, M))
    for l in range(M):
        # Extract the l-th diagonalized matrix.
        Phi_l = D_blocks[:, l*r:(l+1)*r]
        # The diagonal contains the eigenvalues, whose angles are the 1D frequencies.
        diag_elements = np.diag(Phi_l)
        paired_phis[:, l] = np.angle(diag_elements)

    # Phase unwrap along each row.
    unwrapped_phis = np.unwrap(paired_phis, axis=1)

    # Step 3: Least squares to estimate omegas
    omega_hat = 1/delta*(pinv(U_directions) @ unwrapped_phis.T).T
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