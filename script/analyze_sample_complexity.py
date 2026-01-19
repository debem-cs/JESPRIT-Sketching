import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd
from dataset_gen import generate_mixed_poisson_samples
from jesprit import jesprit, compute_error
import time

def analyze_sample_complexity():
    # Configuration
    max_d = 5
    d_range = list(range(1, max_d + 1)) 
    r_range = list(range(1, max_d + 1)) # Will filter valid pairs

    target_rate_error = 0.10           # 10% error threshold (MRE)
    target_weight_error = 0.10         # 10% error threshold (MRE)
    patience = 3                       # Early stopping patience
    
    # Sample sizes to test (logarithmic scale)
    sample_steps = [1000, 5000, 10000, 25000, 50000, 75000, 100000, 150000, 200000]
    
    print(f"Starting Grid Search Sample Complexity Analysis")
    print(f"Target Rate & Weight Error: {target_rate_error}")
    print(f"Dimensions d: {d_range}")
    print(f"Ranks r: {r_range} (where r <= d)")
    print("-" * 50)

    # Ensure log directory exists
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    
    # Data storage for plotting
    results = []

    with open(os.path.join(log_dir, "sample_complexity_grid.txt"), "w") as f:
        f.write("d, r, Samples Needed, Rate Error, Weight Error, Time(s), Success\n")
        f.flush()
        
        for d in d_range:
            for r in r_range:
                # Identification Constraint: typically we need d >= r for simple identifiability,
                # or strictly d >= r. JESPRIT works by finding r-dimensional subspace in d-dim space.
                if r > d:
                    continue
                
                print(f"\nProcessing d={d}, r={r}...")
                
                # 1. Generate Ground Truth
                # To be robust, scale counts slightly with d to keep signal strength comparable?
                # Or just keep random [20, 100].
                A = np.random.randint(20, 100, size=(d, r))
                z = np.eye(r)
                pi = np.random.rand(r)
                pi = pi / np.sum(pi)
                
                lambdas_true = A @ z
                delta = 1.0 / np.max(lambdas_true)
                
                # JESPRIT Parameters
                # Heuristic: M, S, N must be >= r or d.
                # Let's start with generous parameters to isolate sample complexity.
                M = max(d, r) + 10
                S = max(d, r) + 10
                N = max(d, r) + 10
                
                # 2. Find minimum samples
                found_samples = None
                final_rate_err = 1.0
                final_weight_err = 1.0
                final_time = 0.0
                success = False
                
                best_rate_err_pair = float('inf')
                best_samples_pair = None
                prev_rate_err = float('inf')
                increase_count = 0
                
                # Max pool needed
                max_n = sample_steps[-1]
                print(f"  Generating max pool of {max_n} samples...")
                generate_start = time.time()
                try:
                    X_pool, _ = generate_mixed_poisson_samples(A, pi, z, max_n)
                except Exception as e:
                    print(f"  Generation Failed: {e}")
                    results.append({
                        'd': d, 'r': r, 'samples': np.nan, 
                        'rate_err': np.nan, 'time': 0, 'success': False
                    })
                    continue

                for n in sample_steps:
                    print(f"  n={n}...", end="", flush=True)
                    start_time = time.time()
                    
                    try:
                        X_subset = X_pool[:n, :]
                        omega_hat, a_k = jesprit(X_subset, r, M, S, N, delta)
                        elapsed = time.time() - start_time
                        
                        rate_err, weight_err, _, _ = compute_error(lambdas_true, pi, omega_hat, a_k)
                        print(f" Rate:{rate_err:.4f} Wgt:{weight_err:.4f} ({elapsed:.2f}s)")
                        
                        # Track best
                        if rate_err < best_rate_err_pair:
                            best_rate_err_pair = rate_err
                            best_samples_pair = n
                            final_rate_err = rate_err
                            final_weight_err = weight_err
                            final_time = elapsed

                        # Success check
                        if rate_err <= target_rate_error and weight_err <= target_weight_error:
                            found_samples = n
                            success = True
                            final_rate_err = rate_err
                            final_weight_err = weight_err
                            final_time = elapsed
                            print(f"  -> SUCCESS")
                            break
                        
                        # Early stopping
                        if rate_err > prev_rate_err:
                            increase_count += 1
                            if increase_count >= patience:
                                print(f"  -> STOP: Error increased {patience} times.")
                                break
                        else:
                            increase_count = 0
                        prev_rate_err = rate_err
                            
                    except Exception as e:
                        print(f" Failed: {e}")
                        break # Break inner sample loop, move to next d,r
                
                # Finalize result for this (d, r)
                if found_samples is None:
                    # Failed to converge
                    found_samples = max_n * 1.5 # Indicator for plot (saturated)
                    final_rate_err = best_rate_err_pair
                
                # Log to file
                succ_str = "YES" if success else "NO"
                f.write(f"{d}, {r}, {found_samples}, {final_rate_err:.4f}, {final_weight_err:.4f}, {final_time:.4f}, {succ_str}\n")
                f.flush()
                
                results.append({
                    'd': d, 
                    'r': r, 
                    'samples': found_samples if success else max_n, # Cap at max for plotting gradient usually better
                    'raw_samples': found_samples,
                    'rate_err': final_rate_err,
                    'weight_err': final_weight_err,
                    'time': final_time,
                    'success': success
                })

    # --- Plotting ---
    if not results:
        print("No results to plot.")
        return

    df = pd.DataFrame(results)
    
    # 1. Pivot tables
    # Fill missing values with NaN or appropriate indicator
    pivot_samples = df.pivot(index='d', columns='r', values='raw_samples')
    pivot_success = df.pivot(index='d', columns='r', values='success')
    pivot_error = df.pivot(index='d', columns='r', values='rate_err')
    pivot_time = df.pivot(index='d', columns='r', values='time')

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(f'JESPRIT Sample Complexity Analysis (Grid Search)', fontsize=16)

    # Plot 1: Samples Needed
    # Use a mask for where r > d (should be empty/NaN)
    ax = axes[0, 0]
    sns.heatmap(pivot_samples, ax=ax, cmap="viridis", annot=True, fmt=".0f", 
                cbar_kws={'label': 'Min Samples Needed'})
    ax.set_title(f'Minimum Samples for {target_rate_error} Error')
    # Removed invert_yaxis to match other plots (standard matrix orientation)
    
    # Plot 2: Success Status
    ax = axes[0, 1]
    # Convert boolean to int for heatmap: 1=Success, 0=Fail
    sns.heatmap(pivot_success.astype(float), ax=ax, cmap="RdYlGn", annot=True, 
                cbar=False, vmin=0, vmax=1)
    ax.set_title('Success Status (1=Pass, 0=Fail)')

    # Plot 3: Final Rate Error
    ax = axes[1, 0]
    sns.heatmap(pivot_error, ax=ax, cmap="magma_r", annot=True, fmt=".2f", 
                cbar_kws={'label': 'Mean Relative Rate Error'})
    ax.set_title('Lowest Achieved Rate Error')

    # Plot 4: Execution Time
    ax = axes[1, 1]
    sns.heatmap(pivot_time, ax=ax, cmap="coolwarm", annot=True, fmt=".2f", 
                cbar_kws={'label': 'Execution Time (s)'})
    ax.set_title('Execution Time for Winning Run')

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(os.path.join(log_dir, "sample_complexity_grid.png"))
    print(f"\nGrid Search Complete. Plots saved to {log_dir}/sample_complexity_grid.png")

if __name__ == "__main__":
    analyze_sample_complexity()
