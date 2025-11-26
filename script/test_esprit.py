import numpy as np
from esprit import esprit

def test_esprit():
    """
    Test function for the ESPRIT algorithm implementation.
    """
    # 1. Generate synthetic data for a uniform linear array (ULA)
    M = 10      # Number of sensors in the ULA
    N = 200     # Number of snapshots (measurements in time)
    r = 2       # Number of signal sources

    # True frequencies (normalized angular frequencies)
    # These are the phi in the steering vector for a source:
    # a(phi) = [1, exp(j*phi), ..., exp(j*(M-1)*phi)]^T
    phi_true = np.array([0.5, -0.2])
    
    # Steering matrix A, size (M x r)
    m_vector = np.arange(M).reshape(-1, 1)
    A = np.exp(1j * m_vector @ phi_true.reshape(1, -1))

    # Source signals S, size (r x N)
    # We use random complex signals for the sources.
    np.random.seed(0) # for reproducibility
    S = (np.random.randn(r, N) + 1j * np.random.randn(r, N)) / np.sqrt(2)

    # Clean signal matrix X_clean, size (M x N)
    X_clean = A @ S

    # Add complex white Gaussian noise to the signal
    # SNR is approx. 1 / noise_power. Here SNR is 100 (20 dB).
    noise_power = 0.01 
    noise = np.sqrt(noise_power/2) * (np.random.randn(M, N) + 1j * np.random.randn(M, N))
    X = X_clean + noise

    # The esprit() function in src/esprit.py expects the input matrix X
    # to have dimensions (snapshots, sensors), which is N x M.
    # Our generated X is M x N, so we need to transpose it.
    X_for_esprit = X.T

    # 2. Run the ESPRIT algorithm
    # We expect to get 'r' estimated frequencies.
    # esprit returns Psi, we need to compute eigenvalues to get frequencies
    Psi = esprit(X_for_esprit, r)
    eigenvalues, _ = np.linalg.eig(Psi)
    phi_estimated = np.angle(eigenvalues)

    # 3. Compare estimated frequencies with true frequencies
    # We sort both arrays because the order of estimated frequencies is not guaranteed.
    phi_true_sorted = np.sort(phi_true)
    phi_estimated_sorted = np.sort(phi_estimated)

    print("--- ESPRIT Algorithm Test ---")
    print(f"Configuration: {M} sensors, {N} snapshots, {r} sources, SNR ~{10*np.log10(1/noise_power):.0f} dB")
    print(f"True frequencies:      {np.round(phi_true_sorted, 3)}")
    print(f"Estimated frequencies: {np.round(phi_estimated_sorted, 3)}")

    # Assert that the estimated frequencies are close to the true frequencies.
    # A tolerance (atol) is needed because of the noise in the signal.
    try:
        np.testing.assert_allclose(phi_true_sorted, phi_estimated_sorted, atol=1e-2)
        print("\nSUCCESS: Test passed. Estimated frequencies are close to true frequencies.")
    except AssertionError as e:
        print("\nFAILURE: Test failed. Estimated frequencies are not close enough.")
        print(e)

if __name__ == '__main__':
    test_esprit()
