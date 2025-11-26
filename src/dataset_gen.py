import numpy as np

def generate_mixed_poisson_samples(A, pi, z, n_samples):
    """
    Generates samples from a Mixed Poisson Distribution.

    This follows the model:
    1. A component k is chosen with probability pi_k .
    2. The poisson rate vector is calculated: lambda_k = A * z_k .
    3. A count vector X is drawn from Poisson(lambda_k) .

    Parameters
    ----------
    A : np.ndarray
        The (d x m) weight matrix, where d is the number of
        dimensions and m is the number of latent factors .

    pi : np.ndarray
        Array of pi_k probabilities for each of the K 
        components. Must sum to 1.

    z : np.ndarray
        The (m x K) matrix of latent factors .
        Each column 'k' represents the latent vector z_k.

    n_samples : int
        The number of d-dimensional samples to generate (n).

    Returns
    -------
    X : np.ndarray
        The (n_samples x d) matrix of generated counts. Each row
        is one d-dimensional sample vector X_j.
    """

    # --- Input Validation ---
    d, m = A.shape
    if z.shape[0] != m:
        raise ValueError(f"A has {m} columns (m) but z has {z.shape[0]} rows.")

    K = z.shape[1]
    if pi.shape[0] != K:
        raise ValueError(f"z has {K} columns (K) but pi has {pi.shape[0]} elements.")

    if not np.isclose(np.sum(pi), 1.0):
        print(f"Warning: Probabilities 'pi' sum to {np.sum(pi)}. Re-normalizing.")
        pi = pi / np.sum(pi)

    # --- Generation ---

    # 1. calculate all poisson rates 
    Lambda_matrix = A @ z

    # 2. Choose a k latent factor for each of the 'n_samples' at once 
    # This gives an array of k's of lenght 'n_samples' with the correct pi probabilities
    k_indices = np.random.choice(
        np.arange(K), # possible k values: 0 to K-1
        size=n_samples,
        p=pi # probabilities for each component
    )

    # 3. Select the corresponding rate vectors for the chosen k's
    # chosen_rates will be a (d x n_samples) matrix
    chosen_rates = Lambda_matrix[:, k_indices]

    # 4. Generate all Poisson samples at once 
    # np.random.poisson(rates) returns an array of the same shape as 'rates'
    # X_samples will be (d x n_samples)
    X_samples = np.random.poisson(chosen_rates).T

    return X_samples, Lambda_matrix