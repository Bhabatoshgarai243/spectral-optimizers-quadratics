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

# 1. Experimental Setup & Seed (Matched to Original P3)
d, n_cols = 16, 8
iterations = 1400
alpha_old = 0.01143446
eps_large = 3.08

rng = np.random.default_rng(20260912)
eigenvalues = np.array([(1/30) * (30 ** (i / 15)) for i in range(16)])
tr_H = np.sum(eigenvalues)  # Approx 4.79844

A = rng.standard_normal((d, d))
Q, _ = np.linalg.qr(A)
H = Q @ np.diag(eigenvalues) @ Q.T

E_0_raw = rng.standard_normal((d, n_cols))
X_0 = E_0_raw / np.linalg.norm(E_0_raw, ord='fro')

def calc_loss(X):
    return 0.5 * np.trace(X.T @ H @ X)

initial_loss = calc_loss(X_0)

# Quintic Newton-Schulz map (5 stages)
def quintic_ns(Y):
    for _ in range(5):
        YtY = Y.T @ Y
        Y = Y @ (3.4445 * np.eye(n_cols) - 4.7750 * YtY + 2.0315 * (YtY @ YTY if 'YTY' in locals() else YtY @ YtY))
    return Y

# 2. Define the Plotting Runs
methods = [
    'Normalized NS (eps=0)',
    'Normalized NS (eps=3.08)',
    'Exact polar',
    'PolarGrad (u=1.9)',
    'GD'
]

losses = {m: [initial_loss] for m in methods}
update_lengths = {m: [] for m in methods}

# Define the side-runs for the textual reporting
polargrad_u_tests = [1.0, 2.4]
text_report_losses = {u: [initial_loss] for u in polargrad_u_tests}

# 3. Main Optimization Loop
for method in methods:
    X_t = X_0.copy()
    for _ in range(iterations):
        G_t = H @ X_t
        
        # Calculate update steps based on method
        if method == 'Normalized NS (eps=0)':
            update_step = alpha_old * quintic_ns(G_t / np.linalg.norm(G_t, ord='fro'))
            
        elif method == 'Normalized NS (eps=3.08)':
            # Replaces Fixed-scale NS with a large-offset dynamic scale
            g_norm = np.linalg.norm(G_t, ord='fro')
            update_step = alpha_old * quintic_ns(G_t / (g_norm + eps_large))
            
        elif method == 'Exact polar':
            U, _, Vt = np.linalg.svd(G_t, full_matrices=False)
            update_step = alpha_old * (U @ Vt)
            
        elif method == 'PolarGrad (u=1.9)':
            U, S, Vt = np.linalg.svd(G_t, full_matrices=False)
            nuc_norm = np.sum(S)  # Nuclear norm
            alpha_pg = 1.9 / tr_H
            update_step = alpha_pg * nuc_norm * (U @ Vt)
            
        elif method == 'GD':
            update_step = alpha_old * G_t
            
        X_t = X_t - update_step
        
        losses[method].append(calc_loss(X_t))
        update_lengths[method].append(np.linalg.norm(update_step, ord='fro'))

# Run side-tests for u=1.0 and u=2.4 (for text reporting only)
for u in polargrad_u_tests:
    X_t = X_0.copy()
    alpha_pg = u / tr_H
    
    for _ in range(iterations):
        G_t = H @ X_t
        U, S, Vt = np.linalg.svd(G_t, full_matrices=False)
        nuc_norm = np.sum(S)
        update_step = alpha_pg * nuc_norm * (U @ Vt)
        X_t = X_t - update_step
        
        text_report_losses[u].append(calc_loss(X_t))

# Normalize losses relative to initial loss
for m in methods:
    losses[m] = np.array(losses[m]) / initial_loss
for u in polargrad_u_tests:
    text_report_losses[u] = np.array(text_report_losses[u]) / initial_loss

# 4. Print textual reporting values for the text blanks
print("--- Textual Report Values ---")
# -----------------------------------------------------------
print(f"NS (eps=3.08) min value:     {np.min(losses['Normalized NS (eps=3.08)']):.2e}")
print(f"PolarGrad (u=1.0) min value: {np.min(text_report_losses[1.0]):.2e}")
print(f"PolarGrad (u=1.9) min value: {np.min(losses['PolarGrad (u=1.9)']):.2e}")
print(f"PolarGrad (u=2.4) min value: {np.min(text_report_losses[2.4]):.2e}")
# 5. Plotting the 5-Curve Graph
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

styles = {
    'Normalized NS (eps=0)': ('--', 'C3'),
    'Normalized NS (eps=3.08)': ('-', 'C0'),
    'Exact polar': (':', 'C7'),
    'PolarGrad (u=1.9)': ('-', 'C4'),  # New Purple Line
    'GD': ('-.', 'C2')
}

# Top Subplot: Relative Loss
for m in methods:
    ls, color = styles[m]
    ax1.plot(losses[m], linestyle=ls, color=color, label=m, linewidth=2)

ax1.set_yscale('log')
ax1.set_ylabel(r'$f(X_t)/f(X_0)$')
ax1.set_title('(c) Dense-quadratic loss')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xlim(0, iterations)
# ==========================================
# INSET CONTROLS: [x0, y0, width, height]
# x0: left edge (0.0 to 1.0)
# y0: bottom edge (0.0 to 1.0) -> lower values move it into the vacant bottom space
# width, height: dimensions relative to ax1
# ==========================================
inset_x0 = 0.11      # Horizontal position (0.0 to 1.0)
inset_y0 = 0.15      # Vertical position (moved down into empty space)
inset_width = 0.44   # Width relative to ax1
inset_height = 0.28  # Height relative to ax1

axins = ax1.inset_axes([inset_x0, inset_y0, inset_width, inset_height])
steps_300 = np.arange(1200, 1401)
axins.plot(steps_300, losses['Normalized NS (eps=0)'][1200:1401], 
           linestyle=styles['Normalized NS (eps=0)'][0], 
           color=styles['Normalized NS (eps=0)'][1], 
           linewidth=1.2)
axins.plot(steps_300, losses['Exact polar'][1200:1401], 
           linestyle=styles['Exact polar'][0], 
           color=styles['Exact polar'][1], 
           linewidth=1.2)

axins.set_yscale('log')

# Explicitly set only the top and bottom y-ticks
axins.set_yticks([2e-4, 1e-3])
axins.set_yticklabels([r'$2\times 10^{-4}$', r'$10^{-3}$'])

# Hide the intermediate minor ticks automatically generated by log scale
axins.yaxis.set_minor_formatter(plt.NullFormatter())

axins.set_title('last 300 steps', fontsize=12, pad=2)
axins.tick_params(axis='both', which='major', labelsize=10, pad=1) # Changed 'both' to 'major'
axins.grid(True, alpha=0.3, linestyle='--')

# Bottom Subplot: Update Lengths
for m in methods:
    ls, color = styles[m]
    ax2.plot(update_lengths[m], linestyle=ls, color=color, label=m, linewidth=2)

ax2.set_yscale('log')
ax2.set_ylabel(r'$||X_{t+1} - X_t||_F$')
ax2.set_xlabel('Iteration')
ax2.set_title('(d) Update length')
ax2.grid(True, alpha=0.3, linestyle='--')

# Format legend to accommodate 5 items cleanly
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.08))

plt.tight_layout()
plt.savefig('P3_updated_polargrad.pdf', format='pdf', bbox_inches='tight')
plt.show()