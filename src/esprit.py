import numpy as np
from numpy.linalg import svd, pinv, eig

def esprit(X, D):
    """
    Performs ESPRIT on a multi-snapshot measurement matrix X.
    
    Parameters
    ----------
    X : np.ndarray
        n x M measurement matrix.
    D : int
        Number of input signals.

    Returns
    -------
    phi : np.ndarray
          The estimated frequencies.
    Psi : np.ndarray
          Matrix Psi with the same eigenvalues as the rotational invariance matrix.
    """
    # Step 1: Subspace Estimation using Covariance Matrix
    Rxx = 1/X.shape[0] * X.T @ X.conj()
    
    # Get eigenvectors from SVD of covariance matrix.
    U, _, _ = svd(Rxx)
    
    # The signal subspace is the first D eigenvectors
    U_s = U[:, :D]

    # Step 2: Form Rotational Invariance Equation
    U_1 = U_s[:-1, :] # All rows except the last one
    U_2 = U_s[1:, :]  # All rows except the first one

    # The relationship is U_2 ≈ U_1 @ Psi. We solve for Psi.
    Psi = pinv(U_1) @ U_2
    
    """ 
    # COMMENTED OUT TO SAVE COMPUTATION SINCE JESPRIT ONLY REQUIRES Psi 

    # Step 3: Calculate frequencies from the eigenvalues of Psi
    eigenvalues, _ = eig(Psi)
    
    # Frequencies are the angles of the eigenvalues
    phi = np.angle(eigenvalues)
    
    """
    
    return Psi