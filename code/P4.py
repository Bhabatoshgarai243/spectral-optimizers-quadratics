import numpy as np

# 1. Experimental Setup & Seed 
d, n_cols = 16, 8
iterations = 400
fit_window = 200

rng = np.random.default_rng(20260912)
eigenvalues = np.array([(1/30) * (30 ** (i / 15)) for i in range(16)])
A = rng.standard_normal((d, d))
Q, _ = np.linalg.qr(A)
H = Q @ np.diag(eigenvalues) @ Q.T
lambda_max = eigenvalues[-1]  

# Generate unit direction for initial error
E_0_raw = rng.standard_normal((d, n_cols))
E_0_dir = E_0_raw / np.linalg.norm(E_0_raw, ord='fro')

# Quintic Newton-Schulz map (5 stages)
def quintic_ns(Y):
    for _ in range(5):
        YtY = Y.T @ Y
        Y = Y @ (3.4445 * np.eye(n_cols) - 4.7750 * YtY + 2.0315 * (YtY @ YtY))
    return Y

# 2. Define the Test Grid
epsilons = [1e-1, 3e-2, 1e-2]
target_betas = [1.0, 1.9, 2.05]
p_prime_0 = 3.4445 ** 5

def calc_loss(X):
    return 0.5 * np.trace(X.T @ H @ X)

# 3. Run Optimization Loop and Save to File
filename = "P4_results.md"

with open(filename, "w") as f:
    
    # Helper function to print to both console and file
    def write_out(text):
        print(text)
        f.write(text + "\n")
        
    write_out("### P4: Local Rate and Denominator Constant")
    write_out("```text")
    write_out(f"{'Beta':<6} | {'Epsilon':<8} | {'Measured Rate':<15} | {'Theory Rate':<15}")
    write_out("-" * 50)

    results_by_beta = {beta: [] for beta in target_betas}

    for beta_eps in target_betas:
        theory_rate = np.max(np.abs(1 - beta_eps * eigenvalues))
        
        for eps in epsilons:
            alpha = (beta_eps * eps) / (p_prime_0 * lambda_max)
            X_t = E_0_dir * (1e-12 * eps)
            loss_history = []
            
            for t in range(iterations):
                G_t = H @ X_t
                G_norm = np.linalg.norm(G_t, ord='fro')
                update_step = alpha * quintic_ns(G_t / (G_norm + eps))
                X_t = X_t - update_step
                
                if t >= iterations - fit_window:
                    loss_history.append(calc_loss(X_t))
                    
            t_vals = np.arange(iterations - fit_window, iterations)
            log_losses = 0.5 * np.log(loss_history)
            slope, _ = np.polyfit(t_vals, log_losses, 1)
            measured_rate = np.exp(slope)
            
            results_by_beta[beta_eps].append(measured_rate)
            write_out(f"{beta_eps:<6.2f} | {eps:<8.3f} | {measured_rate:<15.6f} | {theory_rate:<15.6f}")

    write_out("\nVerifying consistency across shared Betas:")
    for beta_eps in target_betas:
        rates = results_by_beta[beta_eps]
        spread = max(rates) - min(rates)
        write_out(f"Beta {beta_eps:<4}: Spread between runs = {spread:.2e}")
    write_out("```")

print(f"\n✅ Results successfully saved to {filename}")