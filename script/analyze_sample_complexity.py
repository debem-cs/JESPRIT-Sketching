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
    r_range = list(range(1, max_d + 1)) 
    
    n_trials = 3                       # Number of random trials per (d, r) to compute median
    target_error = 0.10                # 10% error threshold (Average MRE)
    patience = 3                       # Early stopping patience
    
    # Sample sizes to test
    sample_steps = [1000, 5000, 10000, 25000, 50000, 75000, 100000, 150000, 200000]
    max_n = sample_steps[-1]
    
    print(f"Starting Grid Search Sample Complexity Analysis (Median of {n_trials} trials)")
    print(f"Target Average Error: {target_error}")
    print(f"Dimensions d: {d_range}")
    print(f"Ranks r: {r_range} (where r <= d)")
    print("-" * 50)

    # Ensure log directory exists
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    
    # Data storage for plotting
    results = []

    with open(os.path.join(log_dir, "sample_complexity_grid.txt"), "w") as f:
        f.write("d, r, Median Samples, Median Rate Err, Median Weight Err, Median Time, Success Rate\n")
        f.flush()
        
        for d in d_range:
            for r in r_range:
                if r > d:
                    continue
                
                print(f"\nProcessing d={d}, r={r}...")
                
                trial_samples = []
                trial_rate_errs = []
                trial_weight_errs = []
                trial_times = []
                trial_successes = []
                
                for trial_idx in range(n_trials):
                    print(f"  Trial {trial_idx+1}/{n_trials}...", end="", flush=True)
                    
                    # 1. Generate Ground Truth
                    try:
                        A = np.random.randint(20, 100, size=(d, r))
                        z = np.eye(r)
                        pi = np.random.rand(r)
                        pi = pi / np.sum(pi)
                        
                        lambdas_true = A @ z
                        delta = 1.0 / np.max(lambdas_true)
                        
                        # JESPRIT Parameters
                        M = max(d, r) + 10
                        S = max(d, r) + 10
                        N = max(d, r) + 10
                        
                        # Generate Max Pool
                        X_pool, _ = generate_mixed_poisson_samples(A, pi, z, max_n)
                        
                    except Exception as e:
                        print(f" [Setup Failed: {e}]")
                        trial_samples.append(max_n)
                        trial_rate_errs.append(1.0)
                        trial_weight_errs.append(1.0)
                        trial_times.append(0.0)
                        trial_successes.append(False)
                        continue

                    # 2. Find minimum samples for this trial
                    found_n = None
                    final_rate = 1.0
                    final_wgt = 1.0
                    final_t = 0.0
                    success = False
                    
                    best_avg_local = float('inf')
                    prev_avg_local = float('inf')
                    increase_count = 0
                    
                    for n in sample_steps:
                        start_time = time.time()
                        try:
                            X_subset = X_pool[:n, :]
                            omega_hat, a_k = jesprit(X_subset, r, M, S, N, delta)
                            elapsed = time.time() - start_time
                            
                            rate_err, weight_err, _, _ = compute_error(lambdas_true, pi, omega_hat, a_k)
                            avg_err = (rate_err + weight_err) / 2.0
                            
                            # Track best (now based on Average Error)
                            if avg_err < best_avg_local:
                                best_avg_local = avg_err
                                final_rate = rate_err
                                final_wgt = weight_err
                                final_t = elapsed

                            # Success check (Average Error <= Target)
                            if avg_err <= target_error:
                                found_n = n
                                success = True
                                final_rate = rate_err
                                final_wgt = weight_err
                                final_t = elapsed
                                break
                            
                            # Early stopping (on Avg Error)
                            if avg_err > prev_avg_local:
                                increase_count += 1
                                if increase_count >= patience:
                                    break
                            else:
                                increase_count = 0
                            prev_avg_local = avg_err
                            
                        except Exception as e:
                            # print(f" [Failed: {e}]", end="") 
                            break 
                    
                    # Store Trial Result
                    if found_n is None:
                        trial_samples.append(max_n) # Saturated at max (user requested no exaggeration)
                        trial_successes.append(False)
                    else:
                        trial_samples.append(found_n)
                        trial_successes.append(True)
                        
                    trial_rate_errs.append(final_rate)
                    trial_weight_errs.append(final_wgt)
                    trial_times.append(final_t)
                    
                    status = "PASS" if success else "FAIL"
                    print(f" -> {status} (n={found_n if found_n else '>'+str(max_n)})")

                # --- Compute Medians across trials ---
                med_samples = np.median(trial_samples)
                med_rate_err = np.median(trial_rate_errs)
                med_weight_err = np.median(trial_weight_errs)
                med_time = np.median(trial_times)
                success_rate = np.mean(trial_successes)
                
                # Determine overall success:
                # If median samples is saturated, then "Fail". Or based on success rate?
                # Let's say if success rate > 0.5, we consider it a success.
                overall_success = (success_rate >= 0.5)
                
                med_print = f"{med_samples:.0f}"
                if med_samples > max_n:
                    med_print = f">{max_n}"
                
                print(f"  => RESULT: Median Samples={med_print}, Success Rate={success_rate*100:.0f}%")
                
                # Log to file (keep numeric for consistency, or string?)
                # user sees log file too. Let's saturate it at max_n in log or keep as is?
                # Let's keep numeric in CSV for parsing, but maybe cap at max_n for plotting aesthetics?
                # Actually, 1.5*max_n is good for heatmap contrast.
                f.write(f"{d}, {r}, {med_samples}, {med_rate_err:.4f}, {med_weight_err:.4f}, {med_time:.4f}, {success_rate:.2f}\n")
                f.flush()
                
                results.append({
                    'd': d, 
                    'r': r, 
                    'samples': med_samples,
                    'rate_err': med_rate_err,
                    'weight_err': med_weight_err,
                    'avg_err': (med_rate_err + med_weight_err) / 2.0,
                    'time': med_time,
                    'success': overall_success,
                    'success_rate': success_rate
                })

    # --- Plotting ---
    if not results:
        print("No results to plot.")
        return

    df = pd.DataFrame(results)
    
    pivot_samples = df.pivot(index='d', columns='r', values='samples')
    pivot_success = df.pivot(index='d', columns='r', values='success_rate')
    pivot_error = df.pivot(index='d', columns='r', values='avg_err')
    pivot_time = df.pivot(index='d', columns='r', values='time')

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(f'JESPRIT Sample Complexity (Median of {n_trials} Trials)', fontsize=16)

    # Plot 1: Median Samples
    ax = axes[0, 0]
    sns.heatmap(pivot_samples, ax=ax, cmap="viridis", annot=True, fmt=".0f", 
                cbar_kws={'label': 'Samples Needed'})
    ax.set_title(f'Min Samples for {target_error} Avg Error (Rate and Weight)')
    
    # Plot 2: Success Rate
    ax = axes[0, 1]
    sns.heatmap(pivot_success, ax=ax, cmap="RdYlGn", annot=True, fmt=".2f", 
                cbar_kws={'label': 'Success Rate'}, vmin=0, vmax=1)
    ax.set_title(f'Success Rate (over {n_trials} trials)')

    # Plot 3: Average Error
    ax = axes[1, 0]
    sns.heatmap(pivot_error, ax=ax, cmap="magma_r", annot=True, fmt=".2f", 
                cbar_kws={'label': 'Average Error (Rate + Weight)/2'})
    ax.set_title('Avg Error')

    # Plot 4: Median Execution Time
    ax = axes[1, 1]
    sns.heatmap(pivot_time, ax=ax, cmap="coolwarm", annot=True, fmt=".2f", 
                cbar_kws={'label': 'Time (s)'})
    ax.set_title('Execution Time')

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(os.path.join(log_dir, "sample_complexity_grid.png"))
    print(f"\nGrid Search Complete. Plots saved to {log_dir}/sample_complexity_grid.png")

if __name__ == "__main__":
    analyze_sample_complexity()
