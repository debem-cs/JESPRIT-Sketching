import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from jesprit import jesprit
from dataset_gen import generate_mixed_poisson_samples
from pgf import sample_PGF
from esprit import esprit

def check_subspaces():
    # Define Parameters
    A = np.array([
        [100, 1,   1],
        [100, 100, 100],
        [1,   1,   100]
    ])
    z_1 = np.array([[1], [0], [0]]) 
    z_2 = np.array([[0], [1], [0]])
    z_3 = np.array([[0], [0], [1]])
    z = np.hstack([z_1, z_2, z_3]) 
    pi = np.array([0.7, 0.2, 0.1])
    r = np.size(z, 1)
    lambdas_true = A @ z
    
    n_samples = 2000
    delta = 1/np.max(lambdas_true)
    d, m = A.shape
    
    X, _ = generate_mixed_poisson_samples(A, pi, z, n_samples)
    
    M = 5
    S = r + 15
    N = r + 15
    
    all_Z, U_directions, p_base_points = sample_PGF(X, M, S, N, delta)
    
    # Manually run the first steps of JESPRIT to get to ESPRIT
    Z_stack = np.stack(all_Z)
    Z_input = Z_stack.transpose(0, 2, 1)
    
    # Run ESPRIT to get Psi matrices
    # all_Psi: (M, r, r)
    # We need to import esprit properly or copy the logic
    # The script already imports esprit
    
    # Z_input is (M, S, N)
    all_Psi = esprit(Z_input, r)
    
    print("\n--- Checking Simultaneous Diagonalizability of Psi Matrices ---")
    
    Psi0 = all_Psi[0]
    Psi1 = all_Psi[1]
    
    # 1. Commutator Check
    # If they share a basis, they must commute: AB = BA
    commutator = Psi0 @ Psi1 - Psi1 @ Psi0
    comm_norm = np.linalg.norm(commutator)
    
    print(f"\n1. Commutator Norm ||Psi0 Psi1 - Psi1 Psi0||: {comm_norm:.6f}")
    if comm_norm < 1e-5:
        print("   -> Matrices COMMUTE (likely share a basis).")
    else:
        print("   -> Matrices DO NOT COMMUTE (cannot share a basis).")
        
    # 2. Eigenvector Comparison
    # We compute eigenvectors and check if they span the same space or are similar
    _, V0 = np.linalg.eig(Psi0)
    _, V1 = np.linalg.eig(Psi1)
    
    # To compare V0 and V1, we need to handle permutation and scaling.
    # A robust way is to check if V1 diagonalizes Psi0 (approximately)
    # D_approx = inv(V1) @ Psi0 @ V1
    # If V1 is a valid basis for Psi0, D_approx should be diagonal.
    
    V1_inv = np.linalg.pinv(V1)
    D_approx = V1_inv @ Psi0 @ V1
    
    off_diag_energy = np.sum(np.abs(D_approx - np.diag(np.diag(D_approx)))**2)
    total_energy = np.sum(np.abs(D_approx)**2)
    ratio = off_diag_energy / total_energy
    
    print(f"\n2. Cross-Diagonalization Error:")
    print(f"   Can eigenvectors of Psi1 diagonalize Psi0? Off-diagonal energy ratio: {ratio:.6f}")
    
    if ratio < 1e-3:
        print("   -> Eigenvectors are COMPATIBLE.")
    else:
        print("   -> Eigenvectors are INCOMPATIBLE.")

if __name__ == "__main__":
    check_subspaces()
