import numpy as np
# Assuming the test is run from the project root
from pgf import sample_PGF, PGF
from dataset_gen import generate_mixed_poisson_samples

def test_sample_pgf():
    """
    Tests the sample_PGF function to ensure it generates measurement
    matrices with the correct properties and statistical validity.
    """
    print("\n--- Testing PGF Sampling (sample_PGF) ---")

    # --- 1. Setup: Generate synthetic data ---
    d = 3  # Dimensionality
    # Using parameters that yield counts roughly between 10 and 100
    A = np.array([[10, 1], [2, 8], [1, 9]])
    z = np.array([[5, 0.5], [0.1, 10]]).T 
    pi = np.array([0.6, 0.4])
    n_samples = 2000  # Increased sample size for better convergence

    X, _ = generate_mixed_poisson_samples(A, pi, z, n_samples)
    assert X.shape == (n_samples, d)

    # --- 2. Define sampling parameters ---
    M = 5   # Number of directions
    S = 4   # Number of snapshots 
    N = 10  # Number of samples along each line
    delta = 0.01 # Small scale to prevent numerical explosion

    print(f"Parameters: M={M}, S={S}, N={N}, d={d}, delta={delta}")

    # --- 3. Call the function to be tested ---
    all_Z, U_directions, p_base_points_scaled = sample_PGF(X, M, S, N, delta=delta)

    # --- 4. Verification of outputs ---
    print("\n--- Verifying output shapes and types ---")

    # Verify all_Z
    assert isinstance(all_Z, list), "all_Z should be a list"
    assert len(all_Z) == M, f"Length of all_Z should be M={M}"
    expected_Z_shape = (N, S)
    for i, Z_l in enumerate(all_Z):
        assert isinstance(Z_l, np.ndarray), f"Element {i} of all_Z should be a numpy array"
        assert Z_l.shape == expected_Z_shape, f"Shape of Z_{i} is {Z_l.shape}, expected {expected_Z_shape}"
        assert Z_l.dtype == np.complex128, f"Dtype of Z_{i} should be complex128"
    print(f"Verified all_Z: list of {len(all_Z)} matrices of shape {all_Z[0].shape}")

    # Verify U_directions
    assert U_directions.shape == (M, d)
    norms = np.linalg.norm(U_directions, axis=1)
    assert np.allclose(norms, 1.0), "Direction vectors should be normalized"
    print("Verified U_directions normalization.")

    # --- 5. Statistical Verification (Sanity Check) ---
    print("\n--- Verifying PGF values (statistical sanity check) ---")

    # Calculate theoretical rates: lambda_k = A @ z_k
    lambda_1 = A @ z[:, 0]
    lambda_2 = A @ z[:, 1]

    # Generate a test point 'u' with SMALL magnitude.
    # If u is too large (e.g. ~1.0), t^X will explode for X ~ 50.
    u_test = np.random.randn(d) * delta 
    t_test = np.ones(d) + 1j * u_test

    # Theoretical PGF: G(t) = sum(pi_k * exp(<lambda_k, t-1>))
    # Since t = 1 + j*u, then t-1 = j*u.
    term1 = np.exp(1j * np.dot(lambda_1, u_test))
    term2 = np.exp(1j * np.dot(lambda_2, u_test))
    theoretical_pgf = pi[0] * term1 + pi[1] * term2

    # Empirical PGF from the data
    empirical_pgf = PGF(X, t_test)

    print(f"Test u norm: {np.linalg.norm(u_test):.5f}")
    print(f"Theoretical PGF: {theoretical_pgf:.5f}")
    print(f"Empirical PGF:   {empirical_pgf:.5f}")
    
    diff = np.abs(theoretical_pgf - empirical_pgf)
    print(f"Absolute difference: {diff:.5f}")

    # We use a relaxed tolerance because PGF variance is naturally high
    # but with scale=0.01 it should converge reasonably well.
    is_close = np.allclose(theoretical_pgf, empirical_pgf, rtol=0.1, atol=0.05)
    
    if not is_close:
        print("WARNING: Statistical check failed. This might happen due to random noise.")
        print("Ensure 'scale' is small (e.g., 0.01) and 'n_samples' is large.")
    else:
        print("Statistical check passed.")
    
    assert is_close, f"Empirical PGF {empirical_pgf} too far from Theoretical {theoretical_pgf}"

    print("\n--- Test Passed ---")

if __name__ == "__main__":
    test_sample_pgf()