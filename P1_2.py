import numpy as np
import matplotlib.pyplot as plt



plt.rcParams.update({
    'font.size': 14,          # General font size
    'axes.titlesize': 16,     # Subplot title size (e.g., "(c) Dense-quadratic loss")
    'axes.labelsize': 16,     # X and Y axis label size
    'xtick.labelsize': 14,    # X axis tick numbers
    'ytick.labelsize': 14,    # Y axis tick numbers
    'legend.fontsize': 12,    # Legend text size
    'figure.titlesize': 18    # Main figure title size (if used)
})
# ==========================================
# Common Experimental Setup
# ==========================================
d, n = 16, 16
eigenvalues = np.array([(1/30) * (30 ** (i / 15)) for i in range(16)])
H = np.diag(eigenvalues)
X_0 = np.eye(d) / 4

def calc_loss(X):
    return 0.5 * np.trace(X.T @ H @ X)

def spectral_update(G, alpha, h_func):
    U, S, Vt = np.linalg.svd(G, full_matrices=False)
    return alpha * (U @ np.diag(h_func(S)) @ Vt)

# ==========================================
# Run P1 Computations
# ==========================================
iterations_p1 = 1200
alpha_lmax_vals = [1.8, 2.0, 2.2]
results_p1 = {}
initial_loss = calc_loss(X_0)

for val in alpha_lmax_vals:
    alpha = val 
    X_t = X_0.copy()
    loss_history = [initial_loss]
    
    for _ in range(iterations_p1):
        G_t = H @ X_t
        X_t = X_t - spectral_update(G_t, alpha, np.tanh)
        loss_history.append(calc_loss(X_t))
        
    results_p1[val] = np.array(loss_history) / initial_loss

# ==========================================
# Run P2 Computations
# ==========================================
iterations_p2 = 2400
fit_window = 300
alpha_a_vals = np.linspace(0.3, 1.98, 15)

shapings = {
    'GD': (lambda s: s, 1),
    r'$\tanh(\sigma)$': (np.tanh, 1),
    r'$\tanh(4\sigma)$': (lambda s: np.tanh(4 * s), 4),
    r'$\sigma / \sqrt{\sigma^2 + 0.25}$': (lambda s: s / np.sqrt(s**2 + 0.25), 2)
}

results_p2 = {name: [] for name in shapings}

for name, (h_func, a) in shapings.items():
    for alpha_a in alpha_a_vals:
        alpha = alpha_a / a
        X_t = X_0.copy()
        error_norms = []
        
        for t in range(iterations_p2):
            G_t = H @ X_t
            X_t = X_t - spectral_update(G_t, alpha, h_func)
            
            if t >= iterations_p2 - fit_window:
                error_norms.append(np.linalg.norm(X_t, ord='fro'))
                
        t_vals = np.arange(iterations_p2 - fit_window, iterations_p2)
        log_errors = np.log(error_norms)
        slope, _ = np.polyfit(t_vals, log_errors, 1)
        results_p2[name].append(np.exp(slope))

# ==========================================
# Plotting the Combined Figure
# ==========================================
# Set figsize to (8, 5) to match P3
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5))

# Ensure the main figure background is completely transparent
fig.patch.set_alpha(0.0)

# --- Subplot 1 (P1) ---
styles_p1 = {
    1.8: ('-', 'C0', r'$\alpha\lambda_{\max}=1.8$'),
    2.0: ('--', 'C1', r'$\alpha\lambda_{\max}=2.0$'),
    2.2: (':', 'C2', r'$\alpha\lambda_{\max}=2.2$')
}

for val in alpha_lmax_vals:
    linestyle, color, label = styles_p1[val]
    # Set linewidth to 2 to match P3 default
    ax1.plot(results_p1[val], linestyle=linestyle, color=color, label=label, linewidth=2)

ax1.set_yscale('log')
ax1.set_ylim(1e-6, 10) 
ax1.set_xlim(0, 1200)
ax1.set_xlabel('Iteration')
ax1.set_ylabel(r'$f(X_t)/f(X_0)$')
ax1.set_title('(a) Step-size threshold')
ax1.grid(True, alpha=0.3, linestyle='--')

# Clear subplot background
ax1.patch.set_alpha(0.0)
# Default font sizes restored to match P3
ax1.legend(loc='lower left')

# --- Subplot 2 (P2) ---
alpha_a_dense = np.linspace(0.2, 2.05, 200)
theory_rates = [np.max(np.abs(1 - val * eigenvalues)) for val in alpha_a_dense]
ax2.plot(alpha_a_dense, theory_rates, 'k-', linewidth=2, label='Theory', zorder=1)

markers = ['o', 's', '^', '+']
colors = ['C0', 'C1', 'C2', 'C4']
# Restored larger scatter sizes since the figure is now larger
sizes = [140, 90, 50, 20] 

for (name, rates), marker, color, size in zip(results_p2.items(), markers, colors, sizes):
    ax2.scatter(alpha_a_vals, rates, label=name, marker=marker, 
                color=color, s=size, edgecolors='k' if marker != '+' else None, zorder=3)

ax2.set_xlim(0.2, 2.05)
ax2.set_ylim(0.93, 1.01)
ax2.set_xlabel(r'Effective step $\alpha a$')
ax2.set_ylabel(r'Error rate $\rho$')
ax2.set_title('(b) Local rate')
ax2.grid(True, alpha=0.3, linestyle='--')

# Clear subplot background
ax2.patch.set_alpha(0.0)
# Default font sizes restored to match P3
ax2.legend(loc='lower left', ncol=2)

plt.tight_layout()
# Save with transparent=True parameter
plt.savefig('P1_P2_combined.pdf', format='pdf', bbox_inches='tight', transparent=True)
plt.show()