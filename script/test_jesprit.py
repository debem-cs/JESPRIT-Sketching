import numpy as np
from dataset_gen import generate_mixed_poisson_samples
from pgf import sample_PGF
from jesprit import jesprit

def test_jesprit():   
    # --- 1. Define Parameters (Based on PDF Example 1.1) ---
    
    A = np.array([
        [100, 1],  # "calculus"
        [1,  100],  # "football"
        [1,  100]   # "investment"
    ])

    z_1 = np.array([[1], [1]]) 
    z_2 = np.array([[0], [1]])
    
    z = np.hstack([z_1, z_2]) 
    r = np.size(z, 1)

    # Calculate the true lambda rates for each component
    lambdas_true = A @ z

    pi = np.array([0.2, 0.8])
    
    n_samples = 1000
    delta = 1/np.max(lambdas_true)
    
    d, m = A.shape

    print("True component rates (lambda_k):")
    for k in range(lambdas_true.shape[1]):
        print(f"  Component {k+1}:\n{lambdas_true[:, k]}")
    print("\nTrue latent factors (pi):\n", pi)

    X, _ = generate_mixed_poisson_samples(A, pi, z, n_samples)
    
    # Sampling parameters for JESPRIT
    M = d + 5  # Number of directions (>= d)
    S = r + 5  # Number of snapshots (>= r)
    N = r + 5  # Number of samples per line (>= r + 1)
    
    all_Z, U_directions, p_base_points = sample_PGF(X, M, S, N, delta)

    omega_hat, a_k = jesprit(
        all_Z, r, U_directions, p_base_points, delta
    )
    
    print("\n--- JESPRIT Results ---")
    # omega_hat has shape (r, d), so each row is an estimated lambda.
    # We iterate through the rows to print each component's estimated rate vector.
    print("Estimated component rates (lambda_hat):")
    for k in range(omega_hat.shape[0]):
        print(f"  Component {k+1}:\n{omega_hat[k, :]}")
    print("\nEstimated latent factors probabilities (pi_hat):\n", a_k)
    
if __name__ == "__main__":
    test_jesprit()