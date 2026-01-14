import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from jesprit import jesprit, compute_error
from dataset_gen import generate_mixed_poisson_samples
from pgf import sample_PGF

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
    
    n_samples = 5000 # Significant increase to reduce noise
    delta = 1.5/np.max(lambdas_true) # Larger delta for better SNR, still < pi
    d, m = A.shape
    
    print(f"Generating {n_samples} samples...")
    X, _ = generate_mixed_poisson_samples(A, pi, z, n_samples)
    
    M = 50
    S = 30
    N = 30
    
    print("Sampling PGF derivatives...")
    all_Z, U_directions, p_base_points = sample_PGF(X, M, S, N, delta)
    
    print("Running JESPRIT with Global SVD...")
    omega_hat, pi_hat = jesprit(all_Z, r, U_directions, p_base_points, delta)
    
    print("\nEstimated Rates (omega_hat):")
    print(omega_hat)
    
    print("\nTrue Rates (lambdas_true):")
    print(lambdas_true)
    
    print("\nEstimated Weights (pi_hat):")
    print(pi_hat)
    print("\nTrue Weights (pi):")
    print(pi)
    
    rate_error, weight_error = compute_error(lambdas_true, pi, omega_hat, pi_hat)
    print(f"\nRate Error: {rate_error:.4f}")
    print(f"Weight Error: {weight_error:.4f}")
    
    if rate_error > 10.0:
        print("\n[WARNING] Rate error is still high!")
    else:
        print("\n[SUCCESS] Rate error is low.")

if __name__ == "__main__":
    test_global_svd()
