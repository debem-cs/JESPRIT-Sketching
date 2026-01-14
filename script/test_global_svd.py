import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from jesprit_global import jesprit_global
from dataset_gen import generate_mixed_poisson_samples
from pgf import sample_PGF
from jesprit import compute_error

def test_global_svd():
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
    
    M = 20 # Increased M for global SVD stability?
    S = r + 15
    N = r + 15
    
    all_Z, U_directions, p_base_points = sample_PGF(X, M, S, N, delta)
    
    print("Running JESPRIT with Global SVD...")
    omega_hat, _ = jesprit_global(all_Z, r, U_directions, p_base_points, delta)
    
    print("\nEstimated Rates (omega_hat):")
    print(omega_hat)
    
    print("\nTrue Rates (lambdas_true):")
    print(lambdas_true)
    
    rate_error, _ = compute_error(lambdas_true, pi, omega_hat, pi) # Dummy pi_est
    print(f"\nRate Error: {rate_error:.4f}")

if __name__ == "__main__":
    test_global_svd()
