import numpy as np
import matplotlib.pyplot as plt
import os
from dataset_gen import generate_mixed_poisson_samples
from pgf import sample_PGF
from jesprit import jesprit, compute_error

def test_jesprit(num_runs=10):   
    # --- 1. Define Parameters (Based on PDF Example 1.1) ---
    
    # Randomly generate A and pi
    d_dim = 5
    r_dim = 3
    A = np.random.randint(0, 100, size=(d_dim, r_dim))
    
    z = np.eye(r_dim)
    
    pi = np.random.rand(r_dim)
    pi = pi / np.sum(pi)

    r = np.size(z, 1)

    # Calculate the true lambda rates for each component
    lambdas_true = A @ z

    n_samples = 5000
    delta = 1/np.max(lambdas_true)
    
    d, m = A.shape

    print("True component rates (lambda_k):")
    for k in range(lambdas_true.shape[1]):
        print(f"  Component {k+1}:\n{lambdas_true[:, k]}")
    print("\nTrue latent factors (pi):\n", pi)

    X, _ = generate_mixed_poisson_samples(A, pi, z, n_samples)
    
    # Sampling parameters for JESPRIT
    M = d + 15  # Number of directions (>= d)
    S = r + 15  # Number of snapshots (>= r)
    N = r + 15  # Number of samples per line (>= r + 1)
    
    rate_errors = []
    weight_errors = []
    omega_hats = []
    a_ks = []

    print(f"\nRunning JESPRIT {num_runs} times on the same dataset...")

    for i in range(num_runs):
        omega_hat, a_k = jesprit(
            X, r, M, S, N, delta
        )
        
        rate_error, weight_error, omega_aligned, a_k_aligned = compute_error(lambdas_true, pi, omega_hat, a_k)
        rate_errors.append(rate_error)
        weight_errors.append(weight_error)
        omega_hats.append(omega_aligned)
        a_ks.append(a_k_aligned)
        # print(f"Run {i+1}: Rate Error = {rate_error:.4f}, Weight Error = {weight_error:.4f}")

    # --- Analysis & Plotting ---
    avg_rate_error = np.mean(rate_errors)
    std_rate_error = np.std(rate_errors)
    avg_weight_error = np.mean(weight_errors)
    std_weight_error = np.std(weight_errors)

    # Set numpy print options for 2 decimal places
    np.set_printoptions(precision=2, suppress=True)

    print("\n--- Summary Statistics ---")
    print(f"Rate Error: Mean = {avg_rate_error:.2f}, Std = {std_rate_error:.2f}")
    print(f"Weight Error: Mean = {avg_weight_error:.2f}, Std = {std_weight_error:.2f}")

    # Ensure log directory exists
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)

    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Run Index')
    ax1.set_ylabel('Rate Error', color=color)
    ax1.plot(range(1, num_runs + 1), rate_errors, marker='o', linestyle='-', color=color, label='Rate Error')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:orange'
    ax2.set_ylabel('Weight Error', color=color)  # we already handled the x-label with ax1
    ax2.plot(range(1, num_runs + 1), weight_errors, marker='s', linestyle='-', color=color, label='Weight Error')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('JESPRIT Estimation Errors across Runs')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plot_path = os.path.join(log_dir, "test_jesprit.png")
    plt.savefig(plot_path)
    print(f"\nPlot saved to {plot_path}")

    # Save logs
    log_path = os.path.join(log_dir, "test_jesprit.txt")
    with open(log_path, "w") as f:
        f.write("Evaluation Results\n")
        f.write("==================\n\n")
        f.write(f"Default Parameters:\n{{'M': {M}, 'S': {S}, 'N': {N}, 'delta': {delta:.2f}}}\n\n")
        f.write(f"Ground Truth Rates (Lambda):\n{lambdas_true}\n")
        f.write(f"Ground Truth Weights (Pi):\n{pi}\n\n")
        
        for i in range(num_runs):
            f.write(f"--- Run {i+1} ---\n")
            f.write(f"Rate Error: {rate_errors[i]:.2f}\n")
            f.write(f"Weight Error: {weight_errors[i]:.2f}\n")
            f.write(f"Estimated Rates (omega_hat):\n{omega_hats[i]}\n")
            f.write(f"Rate Diff (GT - Est):\n{lambdas_true - omega_hats[i]}\n")
            with np.printoptions(precision=1):
                # Handle potential divide by zero
                rel_err_rate = np.divide(
                    lambdas_true - omega_hats[i], 
                    lambdas_true, 
                    out=np.zeros_like(lambdas_true, dtype=float), 
                    where=lambdas_true!=0
                ) * 100
                f.write(f"Rate Rel Err % ((GT - Est)/GT * 100):\n{rel_err_rate}\n")
            f.write(f"Estimated Weights (a_k):\n{a_ks[i]}\n")
            f.write(f"Weight Diff (GT - Est):\n{pi - a_ks[i]}\n")
            with np.printoptions(precision=1):
                rel_err_weight = np.divide(
                     pi - a_ks[i],
                     pi,
                     out=np.zeros_like(pi, dtype=float),
                     where=pi!=0
                ) * 100
                f.write(f"Weight Rel Err % ((GT - Est)/GT * 100):\n{rel_err_weight}\n")
            f.write("-" * 30 + "\n")
    print(f"Log saved to {log_path}")
    
if __name__ == "__main__":
    test_jesprit(num_runs=1)