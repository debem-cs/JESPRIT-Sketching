import numpy as np
import matplotlib.pyplot as plt
from pgf import sample_PGF
from jesprit import jesprit, compute_error
from dataset_gen import generate_mixed_poisson_samples

def evaluate_parameters():
    # Ground truth parameters
    A = np.array([
        [100, 1,  1],
        [30,  100, 1],
        [1,   1, 100]
    ])

    z = np.eye(A.shape[1]) 
    pi = np.array([0.7, 0.2, 0.1])
    
    d, _ = A.shape
    r = np.size(z, 1)

    n_samples = 5000
    
    # Generate data
    X, lambda_true = generate_mixed_poisson_samples(A, pi, z, n_samples)

    delta = float(1/np.max(lambda_true))

    # Default parameters
    default_params = {
        'M': d + 5,
        'S': r + 5,
        'N': r + 5,
        'delta': delta
    }
    
    # Ranges to test
    param_ranges = {
        'M': np.linspace(d-1, 50, 50-(d-1)+1, dtype=int),
        'S': np.linspace(r-1, 50, 50-(r-1)+1, dtype=int),
        'N': np.linspace(r-1, 50, 50-(r-1)+1, dtype=int),
        'delta': np.linspace(delta/10, 10*delta, 50)
    }

    results = {}
    
    print("Evaluating parameters...")
    
    # Set numpy print options for 2 decimal places
    np.set_printoptions(precision=2, suppress=True)
    
    # Ensure log directory exists
    import os
    if not os.path.exists("log"):
        os.makedirs("log")

    # Open results file
    with open("log/evaluate_parameters.txt", "w") as f:
        f.write("Evaluation Results\n")
        f.write("==================\n\n")
        f.write(f"Default Parameters:\n{default_params}\n\n")
        f.write(f"Ground Truth Rates (Lambda):\n{lambda_true}\n")
        f.write(f"Ground Truth Weights (Pi):\n{pi}\n\n")

        for param_name, values in param_ranges.items():
            print(f"\nTesting {param_name}...")
            f.write(f"--- Testing Parameter: {param_name} ---\n")
            rate_errors = []
            weight_errors = []
            
            for val in values:
                # Update params
                current_params = default_params.copy()
                current_params[param_name] = val
                
                # Run JESPRIT
                try:
                    omega_hat, a_k = jesprit(
                        X,
                        len(pi),
                        current_params['M'],
                        current_params['S'],
                        current_params['N'],
                        current_params['delta']
                    )
                    
                    rate_error, weight_error, omega_aligned, a_k_aligned = compute_error(lambda_true, pi, omega_hat, a_k)
                    rate_errors.append(rate_error)
                    weight_errors.append(weight_error)
                    
                    print(f"  {param_name}={val}: Rate Error={rate_error:.2f}, Weight Error={weight_error:.2f}")
                    
                    # Write to file
                    f.write(f"\nParameter {param_name} = {val}\n")
                    f.write(f"Mean Relative Rate Error: {rate_error:.4f}\n")
                    f.write(f"Mean Relative Weight Error: {weight_error:.4f}\n")
                    f.write(f"Estimated Rates (omega_hat):\n{omega_aligned}\n")
                    f.write(f"Rate Diff (GT - Est):\n{lambda_true - omega_aligned}\n")
                    with np.printoptions(precision=1):
                        rel_err_rate = np.divide(
                            lambda_true - omega_aligned, 
                            lambda_true, 
                            out=np.zeros_like(lambda_true, dtype=float), 
                            where=lambda_true!=0
                        ) * 100
                        f.write(f"Rate Rel Err % ((GT - Est)/GT * 100):\n{rel_err_rate}\n")
                    f.write(f"Estimated Weights (a_k):\n{a_k_aligned}\n")
                    f.write(f"Weight Diff (GT - Est):\n{pi - a_k_aligned}\n")
                    with np.printoptions(precision=1):
                        rel_err_weight = np.divide(
                             pi - a_k_aligned,
                             pi,
                             out=np.zeros_like(pi, dtype=float),
                             where=pi!=0
                        ) * 100
                        f.write(f"Weight Rel Err % ((GT - Est)/GT * 100):\n{rel_err_weight}\n")
                    f.write("-" * 30 + "\n")
                    
                except Exception as e:
                    print(f"  {param_name}={val}: Failed ({e})")
                    f.write(f"\nParameter {param_name} = {val}\n")
                    f.write(f"Failed: {e}\n")
                    f.write("-" * 30 + "\n")
                    rate_errors.append(np.nan)
                    weight_errors.append(np.nan)
                    
            results[param_name] = (values, rate_errors, weight_errors)

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Parameter Evaluation\nDefaults: {default_params}", fontsize=12)
    axes = axes.flatten()
    
    for i, (param_name, (values, rate_errors, weight_errors)) in enumerate(results.items()):
        ax1 = axes[i]
        
        # Plot Rate Error on primary y-axis (left)
        color = 'tab:blue'
        l1 = ax1.plot(values, rate_errors, marker='o', color=color, label='Rate Error')
        ax1.set_xlabel(param_name)
        ax1.set_ylabel("Rate Error", color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_title(f"Effect of {param_name}")
        ax1.grid(True)
        
        # Create secondary y-axis (right) for Weight Error
        ax2 = ax1.twinx()
        color = 'tab:orange'
        l2 = ax2.plot(values, weight_errors, marker='s', color=color, label='Weight Error')
        ax2.set_ylabel("Weight Error", color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        
        # Combine legends
        lns = l1 + l2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc='upper right')
        
    plt.tight_layout()
    plt.subplots_adjust(top=0.90) # Adjust layout to make room for suptitle
    plt.savefig("log/evaluate_parameters.png")
    print("\nEvaluation complete. Results saved to log/evaluate_parameters.png")

if __name__ == "__main__":
    evaluate_parameters()
