import numpy as np
from numpy.linalg import svd, pinv, eig

def esprit(X, D):
    """
    Performs ESPRIT on a multi-snapshot measurement matrix X.
    Supports batch processing if X is a 3D tensor.
    
    Parameters
    ----------
    X : np.ndarray
        Measurement matrix.
        Shape (snapshots, sensors) for single instance.
        Shape (batch, snapshots, sensors) for batch processing.
    D : int
        Number of input signals.

    Returns
    -------
    Psi : np.ndarray
          Matrix Psi with the same eigenvalues as the rotational invariance matrix.
          Shape (r, r) for single instance.
          Shape (batch, r, r) for batch processing.
    """
    # Handle batch dimension
    if X.ndim == 2:
        is_batch = False
        # Add batch dimension for uniform processing: (1, snapshots, sensors)
        X = X[np.newaxis, :, :]
    elif X.ndim == 3:
        is_batch = True
    else:
        raise ValueError("Input X must be 2D or 3D array.")

    # Step 1: Subspace Estimation using Covariance Matrix
    # X: (B, N, M)
    # X.transpose(0, 2, 1): (B, M, N)
    # Rxx: (B, M, M)
    # We use matmul (@) which broadcasts over the batch dimension
    Rxx = 1/X.shape[1] * X.transpose(0, 2, 1) @ X.conj()
    
    # Get eigenvectors from SVD of covariance matrix.
    # svd returns U, S, Vh. U is (B, M, M)
    U, _, _ = svd(Rxx)
    
    # The signal subspace is the first D eigenvectors
    # U_s: (B, M, D)
    U_s = U[:, :, :D]

    # Step 2: Form Rotational Invariance Equation
    # U_1: (B, M-1, D)
    U_1 = U_s[:, :-1, :] 
    # U_2: (B, M-1, D)
    U_2 = U_s[:, 1:, :]

    # The relationship is U_2 ≈ U_1 @ Psi. We solve for Psi.
    # Psi: (B, D, D)
    Psi = pinv(U_1) @ U_2
    
    if not is_batch:
        return Psi[0]
    
    return Psi