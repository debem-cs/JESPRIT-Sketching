import numpy as np
from dataset_gen import generate_mixed_poisson_samples
from pgf import sample_PGF
from jesprit import jesprit

def test_jesprit():   
    # --- 1. Define Parameters (Difficult Case with Overlap) ---
    # Component 1: [100, 100, 1]
    # Component 2: [1, 100, 100]
    # Component 3: [100, 1, 100]
    # Observe that they overlap significantly in pairs.
    
    A = np.eye(3) # Identity mapping for simplicity to control rates directly via z
    
    z_1 = np.array([[30], [100], [1]])
    z_2 = np.array([[1], [100], [100]])
    z_3 = np.array([[100], [1], [100]])
    
    z = np.hstack([z_1, z_2, z_3]) 
    r = np.size(z, 1)

    pi = np.array([0.4, 0.35, 0.25])
    
    n_samples = 5000 # High samples for difficult separation
    delta = 0.5/np.max(z) # Tune delta
    
    d, m = A.shape

    # Calculate the true lambda rates for each component
    lambdas_true = A @ z
    print("True component rates (lambda_k):")
    for k in range(lambdas_true.shape[1]):
        print(f"  Component {k+1}:\n{lambdas_true[:, k]}")
    print("\nTrue latent factors (pi):\n", pi)

    X = generate_mixed_poisson_samples(A, pi, z, n_samples)
    
    # Sampling parameters for JESPRIT
    M = 50  # Number of directions
    S = 30   # Number of snapshots
    N = 30   # Number of samples per line
    
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