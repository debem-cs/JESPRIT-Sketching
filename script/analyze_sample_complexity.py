import numpy as np
import matplotlib.pyplot as plt
import os
from dataset_gen import generate_mixed_poisson_samples
from jesprit import jesprit, compute_error
import time

def analyze_sample_complexity():
    # Configuration
    d_range = np.arange(1, 9, 1)      # Dimensions to test
    target_rate_error = 0.10           # 10% error threshold (MRE)
    target_weight_error = 0.10         # 10% error threshold (MRE)
    patience = 2                       # Number of consecutive error increases before stopping early
    
    # Sample sizes to test (logarithmic scale)
    sample_steps = [1000, 5000, 10000, 25000, 50000, 75000, 100000, 150000, 200000]
    
    samples_needed = []
    
    print(f"Starting Sample Complexity Analysis")
    print(f"Target Rate Error: {target_rate_error}")
    print(f"Target Weight Error: {target_weight_error}")
    print(f"Patience: {patience} consecutive increases")
    print(f"Dimensions to test: {d_range}")
    print("-" * 50)

    # Ensure log directory exists
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    
    with open(os.path.join(log_dir, "sample_complexity.txt"), "w") as f:
        f.write("Dimension (d), Rank (r=d-3), Samples Needed, Rate Error, Weight Error, Time(s)\n")
        f.flush()
        
        for d in d_range:
            r = d - 3
            if r < 1:
                print(f"Skipping d={d} because r={r} < 1")
                continue
                
            print(f"\nTesting Dimension d={d}, r={r}...")
            
            # 1. Generate a fixed Ground Truth Model (A, pi) for this dimension
            # Scale range slightly to avoid zeros but keep counts reasonable
            A = np.random.randint(20, 100, size=(d, r))
            z = np.eye(r)
            pi = np.random.rand(r)
            pi = pi / np.sum(pi)
            
            lambdas_true = A @ z
            delta = 1.0 / np.max(lambdas_true)
            
            # JESPRIT Parameters (Heuristic scaling with d)
            M = d + 10
            S = r + 10
            N = r + 10
            
            # 2. Find minimum samples required
            found_samples = None
            final_rate_err = None
            final_weight_err = None
            final_time = None
            
            best_rate_err_for_d = float('inf')
            best_samples_for_d = None
            prev_rate_err = float('inf')
            increase_count = 0
            
            # Generate a large pool of data first to be efficient?
            # Max samples needed is sample_steps[-1]
            max_n = sample_steps[-1]
            print(f"  Generating max pool of {max_n} samples...")
            X_pool, _ = generate_mixed_poisson_samples(A, pi, z, max_n)
            
            for n in sample_steps:
                print(f"  Checking params with n={n}...", end="", flush=True)
                start_time = time.time()
                
                # Slice data
                X_subset = X_pool[:n, :]
                
                try:
                    # Run JESPRIT
                    omega_hat, a_k = jesprit(X_subset, r, M, S, N, delta)
                    
                    elapsed = time.time() - start_time
                    
                    # Compute Error
                    rate_err, weight_err, _, _ = compute_error(lambdas_true, pi, omega_hat, a_k)
                    
                    print(f" Rate Err: {rate_err:.4f}, Weight Err: {weight_err:.4f} ({elapsed:.2f}s)")
                    
                    # Update best (prioritize rate error for "best" tracking)
                    if rate_err < best_rate_err_for_d:
                        best_rate_err_for_d = rate_err
                        best_samples_for_d = n
                        final_weight_err = weight_err
                        final_time = elapsed
                    
                    # Success Condition (AND operation)
                    if rate_err <= target_rate_error and weight_err <= target_weight_error:
                        found_samples = n
                        final_rate_err = rate_err
                        final_weight_err = weight_err
                        break
                    
                    # Early Stopping Condition
                    if rate_err > prev_rate_err:
                        increase_count += 1
                        if increase_count >= patience:
                            print(f"  -> Stopping early: Rate Error increased {patience} times consecutively. Using best found: {best_samples_for_d} (Rate Err: {best_rate_err_for_d:.4f})")
                            found_samples = best_samples_for_d
                            final_rate_err = best_rate_err_for_d
                            break
                    else:
                        increase_count = 0 # Reset patience
                        
                    prev_rate_err = rate_err
                        
                except Exception as e:
                    print(f" Failed: {e}")
            
            # Record result
            if found_samples:
                pass # Already set
            else:
                found_samples = best_samples_for_d
                final_rate_err = best_rate_err_for_d
            
            # Sanity check
            if found_samples is None:
                found_samples = max_n 
                final_rate_err = 1.0
                final_weight_err = 1.0
                final_time = 0.0

            if final_rate_err <= target_rate_error and final_weight_err <= target_weight_error:
                 print(f"  -> SUCCESS: Requires {found_samples} samples.")
            else:
                 print(f"  -> WARNING: Did not reach targets. Best Rate Err: {final_rate_err:.4f}, Best Weight Err: {final_weight_err:.4f}")

            samples_needed.append(found_samples)
            f.write(f"{d}, {r}, {found_samples}, {final_rate_err:.4f}, {final_weight_err:.4f}, {final_time:.4f}\n")
            f.flush()

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(d_range, samples_needed, marker='o', linewidth=2)
    plt.xlabel("Dimension (d)")
    plt.ylabel("Required Samples")
    plt.title(f"Sample Complexity of JESPRIT\nTarget Rate & Weight Error <= {target_rate_error*100}%")
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    # Check if we hit the ceiling
    max_step = sample_steps[-1]
    if any(s > max_step for s in samples_needed):
        plt.axhline(y=max_step, color='r', linestyle='--', label='Max Tested Samples')
        plt.legend()
        
    plt.tight_layout()
    plot_path = os.path.join(log_dir, "sample_complexity.png")
    plt.savefig(plot_path)
    print(f"\nAnalysis Complete. Plot saved to {plot_path}")

if __name__ == "__main__":
    analyze_sample_complexity()
