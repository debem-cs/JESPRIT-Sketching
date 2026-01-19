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
    max_d = 10
    d_range = list(range(1, max_d + 1)) 
    r_range = list(range(1, max_d + 1)) 
    
    n_trials = 7                       # Number of random trials per (d, r) to compute avg
    target_error = 0.10                # 10% error threshold (Average MRE)
    patience = 2                       # Early stopping patience
    
    # Sample sizes to test
    sample_steps = [1000, 10000, 50000, 100000]
    max_n = sample_steps[-1]
    
    print(f"Starting Grid Search Sample Complexity Analysis (avg of {n_trials} trials)")
    print(f"Target Average Error: {target_error}")
    print(f"Dimensions d: {d_range}")
    print(f"Ranks r: {r_range} (where r <= d)")
    print("-" * 50)

    # Ensure log directory exists
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    
    # Data storage for plotting
    results = []

    MAX_CONSECUTIVE_FAILURES = 1
    success_threshold = 0.7   # Threshold for success rate (fraction of trials passed)
    
    with open(os.path.join(log_dir, "sample_complexity.txt"), "w") as f:
        f.write("d, r, Trial, Samples, Rate Err, Weight Err, Time, Success\n")
        f.flush()

        
        for d in d_range:
            consecutive_failures = 0
            for r in r_range:
                # Removed the constraint r > d restriction as per user request
                # if r > d:
                #    continue
                
                # Check for early stopping on r-loop if we are failing too much
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    print(f"Skipping remaining r values for d={d} due to {consecutive_failures} consecutive failures.")
                    break

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
                        A = np.random.randint(0, 100, size=(d, r))
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
                            
                        # Log intermediate result (every sample size checked)
                            success_step = (avg_err <= target_error)
                            f.write(f"{d}, {r}, {trial_idx+1}, {n}, {rate_err:.4f}, {weight_err:.4f}, {elapsed:.4f}, {int(success_step)}\n")
                            f.flush()

                            # Track best (now based on Average Error)
                            if avg_err < best_avg_local:
                                best_avg_local = avg_err
                                final_rate = rate_err
                                final_wgt = weight_err
                                final_t = elapsed

                            # Success check (Average Error <= Target)
                            if success_step:
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

                # --- Compute Stats (Average of Successful Runs) ---
                success_rate = np.mean(trial_successes)
                
                # Determine overall success based on threshold
                overall_success = (success_rate >= success_threshold)
                
                # Filter indices of successful runs
                success_indices = [i for i, s in enumerate(trial_successes) if s]
                
                if success_indices:
                    # Compute average of metrics for SUCCESSFUL runs only
                    avg_samples = np.mean([trial_samples[i] for i in success_indices])
                    avg_rate_err = np.mean([trial_rate_errs[i] for i in success_indices])
                    avg_weight_err = np.mean([trial_weight_errs[i] for i in success_indices])
                    avg_time = np.mean([trial_times[i] for i in success_indices])
                else:
                    # Fallback if NO runs succeeded (average of all failed runs)
                    avg_samples = np.mean(trial_samples)
                    avg_rate_err = np.mean(trial_rate_errs)
                    avg_weight_err = np.mean(trial_weight_errs)
                    avg_time = np.mean(trial_times)
                
                # Update consecutive failures counter for early r-loop stopping
                if overall_success:
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                
                avg_print = f"{avg_samples:.0f}"
                if avg_samples > max_n:
                    avg_print = f">{max_n}"
                
                print(f"  => RESULT: Avg Samples(Succ)={avg_print}, Success Rate={success_rate*100:.0f}%")
                
                # Log summary Row
                f.write(f"{d}, {r}, Average, {avg_samples}, {avg_rate_err:.4f}, {avg_weight_err:.4f}, {avg_time:.4f}, {success_rate:.2f}\n")
                f.flush()
                
                results.append({
                    'd': d, 
                    'r': r, 
                    'samples': avg_samples,
                    'rate_err': avg_rate_err,
                    'weight_err': avg_weight_err,
                    'avg_err': (avg_rate_err + avg_weight_err) / 2.0,
                    'time': avg_time,
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
    fig.suptitle(f'JESPRIT Sample Complexity (Avg of Successful Runs over {n_trials} Trials)', fontsize=16)

    # Plot 1: Avg Samples
    ax = axes[0, 0]
    sns.heatmap(pivot_samples, ax=ax, cmap="viridis", annot=True, fmt=".0f", 
                cbar_kws={'label': 'Avg Samples (Success)'})
    ax.set_title(f'Avg Samples for {target_error} Error')
    
    # Plot 2: Success Rate
    ax = axes[0, 1]
    sns.heatmap(pivot_success, ax=ax, cmap="RdYlGn", annot=True, fmt=".2f", 
                cbar_kws={'label': 'Success Rate'}, vmin=0, vmax=1)
    ax.set_title(f'Success Rate (over {n_trials} trials)')

    # Plot 3: Average Error
    ax = axes[1, 0]
    sns.heatmap(pivot_error, ax=ax, cmap="magma_r", annot=True, fmt=".2f", 
                cbar_kws={'label': 'Avg Error (Success Only)'})
    ax.set_title('Avg Error (Rate + Weight)/2')

    # Plot 4: Avg Execution Time
    ax = axes[1, 1]
    sns.heatmap(pivot_time, ax=ax, cmap="coolwarm", annot=True, fmt=".2f", 
                cbar_kws={'label': 'Avg Time (s)'})
    ax.set_title('Avg Execution Time (Success Only)')

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(os.path.join(log_dir, "sample_complexity.png"))
    print(f"\nGrid Search Complete. Plots saved to {log_dir}/sample_complexity.png")

if __name__ == "__main__":
    analyze_sample_complexity()
