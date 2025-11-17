import numpy as np
from numpy.linalg import eig

def joint_diag(A, jthresh=1e-8):
    """
    Joint approximate diagonalization of n (complex) matrices.

    This function is a direct Python/NumPy port of the MATLAB function
    by Jean-Francois Cardoso (https://www2.iap.fr/users/cardoso/jointdiag.html).

    The algorithm finds a unitary matrix V such that the matrices
    V' * A_i * V are as diagonal as possible.

    Parameters
    ----------
    A : np.ndarray
        A (m, n*m) complex numpy array, representing the horizontal
        concatenation of n matrices of size (m, m).
        A = [A1, A2, ..., An]

    jthresh : float, optional
        The threshold for stopping the Jacobi rotations. The algorithm
        stops when all Givens rotation sines are smaller than this
        threshold. Default is 1e-8.

    Returns
    -------
    V : np.ndarray
        The (m, m) unitary matrix that jointly diagonalizes the
        set of matrices.

    D : np.ndarray
        The (m, n*m) array of jointly diagonalized matrices,
        D = [V'*A1*V, ..., V'*An*V].
    """

    # Constants used in the algorithm, defined once at the module level
    B = np.array([[1, 0, 0], [0, 1, 1], [0, -1j, 1j]], dtype=complex)
    Bt = B.T.conj()

    # Get dimensions
    m, nm = A.shape
    
    if nm % m != 0:
        raise ValueError("The number of columns in A must be a multiple of the number of rows.")
        
    n = nm // m

    # Make a copy to not overwrite the original data
    A = A.copy()

    # Initialize V to the identity matrix
    # We use complex type because V will be multiplied by complex G
    V = np.eye(m, dtype=complex)
    encore = True

    while encore:
        encore = False  # Will be set to True if a rotation is performed

        for p in range(m - 1):
            for q in range(p + 1, m):

                # Get the column indices for all n matrices
                # A(p, Ip) in MATLAB selects the (p,p) element of all n matrices
                # A(p, Iq) selects (p,q) of all n matrices
                # A(q, Ip) selects (q,p) of all n matrices
                Ip = np.arange(p, nm, m)
                Iq = np.arange(q, nm, m)

                # Compute the Givens angles
                # g is a 3-by-n matrix
                g = np.vstack([
                    A[p, Ip] - A[q, Iq],
                    A[p, Iq],
                    A[q, Ip]
                ])
                
                # Compute the 3x3 matrix for eigenvalue decomposition
                # T_matrix = real(B * (g * g') * B') in MATLAB
                gg_hermitian = g @ g.T.conj()
                T_matrix = np.real(B @ gg_hermitian @ Bt)

                # Eigenvalue decomposition
                # MATLAB [vcp, D] = eig(X) -> D is matrix, V are eigenvectors
                # NumPy  D, vcp = eig(X) -> D is vector, V are eigenvectors
                D_eigvals, vcp = eig(T_matrix)

                # Sort eigenvalues and get eigenvector for the largest one
                # K(3) in MATLAB is the last one (1-based index)
                K = np.argsort(D_eigvals)
                angles = vcp[:, K[-1]] # K[-1] is the index of the largest eigenvalue

                # Fix sign ambiguity
                if angles[0] < 0:
                    angles = -angles

                # Compute sine and cosine
                c = np.sqrt(0.5 + angles[0] / 2.0)
                # s = 0.5 * (angles(2) - j*angles(3)) / c in MATLAB
                s = 0.5 * (angles[1] - 1j * angles[2]) / c

                if np.abs(s) > jthresh:
                    # A rotation is needed
                    encore = True

                    # G is the 2x2 Givens rotation matrix
                    G = np.array([[c, -np.conj(s)],
                                  [s, c]], dtype=complex)

                    # Update V
                    pair = np.array([p, q])
                    V[:, pair] = V[:, pair] @ G

                    # Update A rows
                    # A(pair,:) = G' * A(pair,:) in MATLAB
                    A[pair, :] = G.T.conj() @ A[pair, :]
                    
                    # Update A columns
                    # This is the tricky part:
                    # A(:,[Ip Iq]) = [ c*A(:,Ip)+s*A(:,Iq) -conj(s)*A(:,Ip)+c*A(:,Iq) ]
                    # This must be done simultaneously, so we cache the old values
                    
                    A_p_cols = A[:, Ip]
                    A_q_cols = A[:, Iq]
                    
                    A[:, Ip] = c * A_p_cols + s * A_q_cols
                    A[:, Iq] = -np.conj(s) * A_p_cols + c * A_q_cols

    # The diagonalized matrices
    D = A
    return V, D