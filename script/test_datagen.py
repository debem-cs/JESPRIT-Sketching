import numpy as np
from dataset_gen import generate_mixed_poisson_samples

def test_generation():
    """
    Tests the generate_mixed_poisson_samples function.
    """
    print("--- Testing Dataset Generation ---")

    # --- 1. Define Parameters (Based on PDF Example 1.1) ---
    
    # d=3 dimensions (words), m=2 latent factors (topics) [source: 21-31]
    # (d x m) matrix [source: 35]
    A = np.array([
        [10, 1],  # "calculus"
        [2,  8],  # "football"
        [1,  9]   # "investment"
    ])
    
    # K=2 components (Academic, Sports & Finance)
    # We take z_1 and z_2 from the PDF's scenarios [source: 46, 57]
    
    # z_1: "Academic" (m=2, K=1)
    z_1 = np.array([[5], [0.1]]) 
    
    # z_2: "Sports & Finance" (m=2, K=1)
    z_2 = np.array([[0.5], [10]])
    
    # z: (m x K) matrix. Each column is a z_k [source: 83]
    z = np.hstack([z_1, z_2]) # (2 x 2) matrix
    
    # pi: (K,) array of probabilities [source: 83]
    # Let's assume a 60% chance of "Academic" and 40% of "Sports & Finance"
    pi = np.array([0.6, 0.4])
    
    # n_samples: Number of samples to generate
    n_samples = 1000
    
    d, m = A.shape
    K = z.shape[1]
    
    print(f"Parameters: d={d}, m={m}, K={K}, n_samples={n_samples}")
    print("A matrix (d x m):\n", A)
    print("z matrix (m x K):\n", z)
    print("pi vector (K,):\n", pi)

    # --- 2. Call the Function ---
    X, _ = generate_mixed_poisson_samples(A, pi, z, n_samples)

    # --- 3. Verify Output ---
    print("\n--- Verification ---")
    
    # Check shape
    expected_shape = (n_samples, d)
    print(f"Generated X shape: {X.shape} (Expected: {expected_shape})")
    assert X.shape == expected_shape, "Shape mismatch!"
    
    # Print first 5 samples
    print("First 5 samples of X (n_samples x d):\n", X[:5, :])
    
    # --- 4. Statistical Verification (Compare Means) ---
    
    # Calculate theoretical rates for each component
    # lambda_k = A * z_k
    lambda_1 = A @ z_1 # (3 x 1)
    lambda_2 = A @ z_2 # (3 x 1)
    
    print("\nTheoretical Rates (Means):")
    print(f"lambda_1 (Academic):\n {lambda_1.flatten()}")
    print(f"lambda_2 (Sports/Finance):\n {lambda_2.flatten()}")

    # The theoretical expected mean of the *mixture* is:
    # E[X] = pi_1 * lambda_1 + pi_2 * lambda_2
    expected_mean = (pi[0] * lambda_1) + (pi[1] * lambda_2)
    
    # The actual mean from our generated samples is:
    actual_mean = np.mean(X, axis=0) # axis=0 finds mean down the columns

    print("\nMean Verification:")
    print("Theoretical Expected Mean (E[X]):\n", expected_mean.flatten())
    print("Actual Sample Mean (from 1000 samples):\n", actual_mean)
    
    # Check if they are close
    is_close = np.allclose(expected_mean.flatten(), actual_mean, rtol=0.1)
    print(f"\nMeans are close (within 10%): {is_close}")
    assert is_close, "Actual mean is too far from theoretical mean!"
    
    print("\n--- Test Passed ---")


if __name__ == "__main__":
    test_generation()