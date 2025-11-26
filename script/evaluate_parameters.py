import numpy as np
import matplotlib.pyplot as plt
from pgf import sample_PGF
from jesprit import jesprit
from dataset_gen import generate_mixed_poisson_samples
import itertools

def compute_error(lambda_true, pi_true, lambda_est, pi_est):
    """
    Computes the estimation error, handling permutation ambiguity.
    Returns (rate_error, weight_error) for the best permutation.
    """
    r = len(pi_true)
    # Try all permutations to match estimated components to true components
    import itertools
    permutations = list(itertools.permutations(range(r)))
    
    min_total_error = float('inf')
    best_rate_error = 0.0
    best_weight_error = 0.0
    
    for perm in permutations:
        perm = list(perm)
        lambda_est_perm = lambda_est[perm, :].T
        
        # Error in rates (normalized)
        rate_error = np.linalg.norm(lambda_true - lambda_est_perm) / np.linalg.norm(lambda_true)
        
        # Error in weights
        weight_error = np.linalg.norm(pi_true - pi_est[perm])
        
        total_error = rate_error + weight_error
        
        if total_error < min_total_error:
            min_total_error = total_error
            best_rate_error = rate_error
            best_weight_error = weight_error
            
    return best_rate_error, best_weight_error

def evaluate_parameters():
    # Ground truth parameters
    A = np.array([
        [100, 1],
        [1,  100]
    ])
    z_1 = np.array([[1], [0]]) 
    z_2 = np.array([[0], [1]])
    z = np.hstack([z_1, z_2]) 
    pi = np.array([0.2, 0.8])
    
    d, _ = A.shape
    r = np.size(z, 1)

    n_samples = 1000
    
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
    # Define start, stop, step for each parameter
    param_ranges = {
        'M': np.arange(1, 20, 1),      # 5, 10, ..., 50
        'S': np.arange(1, 20, 1),      # 5, 10, ..., 50
        'N': np.arange(1, 20, 1),      # 5, 10, ..., 50
        'delta': np.arange(0.0005, 0.02, 0.0005) # 0.005, 0.01, ..., 0.1
    }
    
    results = {}
    
    print("Evaluating parameters...")
    
    # Ensure log directory exists
    import os
    if not os.path.exists("log"):
        os.makedirs("log")

    # Open results file
    with open("log/parameter_evaluation.txt", "w") as f:
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
                    all_Z, U_directions, p_base_points = sample_PGF(
                        X, 
                        current_params['M'], 
                        current_params['S'], 
                        current_params['N'], 
                        current_params['delta']
                    )
                    
                    omega_hat, a_k = jesprit(
                        all_Z, 
                        len(pi), 
                        U_directions, 
                        p_base_points, 
                        current_params['delta']
                    )
                    
                    rate_error, weight_error = compute_error(lambda_true, pi, omega_hat, a_k)
                    rate_errors.append(rate_error)
                    weight_errors.append(weight_error)
                    
                    print(f"  {param_name}={val}: Rate Error={rate_error:.4f}, Weight Error={weight_error:.4f}")
                    
                    # Write to file
                    f.write(f"\nParameter {param_name} = {val}\n")
                    f.write(f"Rate Error: {rate_error:.6f}\n")
                    f.write(f"Weight Error: {weight_error:.6f}\n")
                    f.write(f"Estimated Rates (omega_hat):\n{omega_hat}\n")
                    f.write(f"Estimated Weights (a_k):\n{a_k}\n")
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
        ax = axes[i]
        ax.plot(values, rate_errors, marker='o', label='Rate Error')
        ax.plot(values, weight_errors, marker='s', label='Weight Error')
        ax.set_title(f"Effect of {param_name}")
        ax.set_xlabel(param_name)
        ax.set_ylabel("Estimation Error")
        ax.legend()
        ax.grid(True)
        
    plt.tight_layout()
    plt.subplots_adjust(top=0.90) # Adjust layout to make room for suptitle
    plt.savefig("log/parameter_evaluation.png")
    print("\nEvaluation complete. Results saved to log/parameter_evaluation.png")

if __name__ == "__main__":
    evaluate_parameters()
