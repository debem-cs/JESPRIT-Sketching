import numpy as np
from joint_diag import joint_diag

def test_joint_diag():
    """
    Tests the joint_diag function by creating a set of matrices that are
    known to be jointly diagonalizable and verifying the output.
    """
    print("\n--- Testing Joint Diagonalization (joint_diag) ---")

    # --- 1. Setup: Create a known problem ---
    m = 5  # Dimension of the matrices
    n = 10 # Number of matrices to diagonalize

    print(f"Parameters: m={m}, n={n}")

    # Create a known random unitary matrix V
    # We do this by taking the 'Q' from a QR decomposition of a random matrix
    Z = np.random.randn(m, m) + 1j * np.random.randn(m, m)
    Q, _ = np.linalg.qr(Z)
    V_true = Q

    # Create a set of 'n' random diagonal matrices
    all_D_true = []
    for _ in range(n):
        diag_elements = np.random.randn(m) + 1j * np.random.randn(m)
        D_true = np.diag(diag_elements)
        all_D_true.append(D_true)

    # Create the input matrices A_k = V * D_k * V'
    all_A = []
    for D_k in all_D_true:
        A_k = V_true @ D_k @ V_true.T.conj()
        all_A.append(A_k)

    # Concatenate the matrices horizontally to create the input for joint_diag
    A_stacked = np.hstack(all_A)
    assert A_stacked.shape == (m, n * m)

    # --- 2. Call the function to be tested ---
    V_hat, D_hat_stacked = joint_diag(A_stacked, jthresh=1e-9)

    # --- 3. Verification ---
    print("\n--- Verifying output properties ---")

    # Verify V_hat is unitary (V_hat * V_hat' = I)
    identity_matrix = np.eye(m, dtype=complex)
    V_product = V_hat @ V_hat.T.conj()
    assert np.allclose(V_product, identity_matrix), "V_hat is not a unitary matrix."
    print("Verified that the output matrix V_hat is unitary.")

    # Verify that the output matrices are diagonal
    # We do this by checking that the energy of the off-diagonal elements is
    # much smaller than the energy of the diagonal elements.
    off_diag_energy = 0
    diag_energy = 0

    for i in range(n):
        D_hat_k = D_hat_stacked[:, i*m:(i+1)*m]
        diag_part = np.diag(np.diag(D_hat_k))
        off_diag_part = D_hat_k - diag_part
        
        off_diag_energy += np.sum(np.abs(off_diag_part)**2)
        diag_energy += np.sum(np.abs(diag_part)**2)

    diagonality_ratio = off_diag_energy / diag_energy
    assert diagonality_ratio < 1e-12, f"Matrices are not diagonal. Ratio={diagonality_ratio}"
    print(f"Verified that output matrices are diagonal (off-diag/diag energy ratio: {diagonality_ratio:.2e}).")
    
    print("\n--- Test Passed ---")

if __name__ == "__main__":
    test_joint_diag()
