import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# --- Configuration ---
# Use a visually appealing style for the plot
plt.style.use('seaborn-v0_8-whitegrid')
# Set a consistent figure size for three plots
FIG_SIZE = (12, 14) # Increased height for the third plot
TICK_FONT_SIZE = 12
LEGEND_FONT_SIZE = 12

# --- Color Palette (from your blog's theme) ---
color_palette = {
    "primary": "#0A58CA",   # Strong Blue
    "secondary": "#6A7C92",  # Slate Gray
    "accent": "#6F42C1",     # Digital Violet
    "background": "#F8F9FA", # Alabaster White
    "text": "#212529",       # Dark Charcoal
}

def stable_log_softmax(x):
    """Computes log(softmax(x)) in a numerically stable way."""
    # Subtract the max for numerical stability (the log-sum-exp trick)
    x_max = np.max(x, axis=-1, keepdims=True)
    log_sum_exp = x_max + np.log(np.sum(np.exp(x - x_max), axis=-1, keepdims=True))
    return x - log_sum_exp

# --- Potentially Unstable Helper Functions ---
def unstable_softmax(x):
    """Computes softmax(x), which can underflow for large negative logits."""
    e_x = np.exp(x)
    sum_e_x = np.sum(e_x, axis=-1, keepdims=True)
    # This division can become 0/1=0 if the numerator underflows
    return e_x / sum_e_x

# --- Data Simulation ---
# Increased points for a smoother curve on the zoomed-in plot
negative_logit_magnitudes = np.linspace(0, 800, 1000)
true_ratio = 1.2

# Logits for 2 actions: [action1, action2]
logits_old = np.array([[-m, 0] for m in negative_logit_magnitudes])
logits_new = np.array([[-m + np.log(true_ratio), 0] for m in negative_logit_magnitudes])

# --- Calculation Methods ---

# 1. Stable Log-Probability Method
log_probs_old = stable_log_softmax(logits_old)[:, 0]
log_probs_new = stable_log_softmax(logits_new)[:, 0]
stable_ratios = np.exp(log_probs_new - log_probs_old)

# 2. Unstable Direct Ratio Method
probs_old = unstable_softmax(logits_old)[:, 0]
probs_new = unstable_softmax(logits_new)[:, 0]
with np.errstate(divide='ignore', invalid='ignore'):
    unstable_ratios = probs_new / probs_old

# --- Error Calculation for the new plot ---
stable_error = stable_ratios - true_ratio
unstable_error = unstable_ratios - true_ratio

# --- Visualization ---
fig, axes = plt.subplots(2, 1, figsize=FIG_SIZE, sharex=True)
fig.set_facecolor(color_palette["background"])

# --- Subplot 1: Stable Log-Probability Method ---
ax1 = axes[0]
ax1.set_facecolor(color_palette["background"])
ax1.set_title(
    'exp(log_softmax(new) - log_softmax(old))',
    color=color_palette["text"], fontsize=16, weight='bold'
)
ax1.plot(
    negative_logit_magnitudes, stable_ratios,
    color=color_palette["accent"],
    linewidth=2
)
ax1.axhline(
    y=true_ratio,
    color=color_palette["secondary"],
    linewidth=1,
    linestyle='--'
)
ax1.set_ylabel('Calculated Ratio', color=color_palette["text"], fontsize=12)
ax1.set_ylim(0, true_ratio + 0.5)
ax1.tick_params(colors=color_palette["text"], labelsize=TICK_FONT_SIZE)
for spine in ax1.spines.values():
    spine.set_edgecolor(color_palette["secondary"])

# --- Subplot 2: Unstable Direct Ratio Method ---
ax2 = axes[1]
ax2.set_facecolor(color_palette["background"])
ax2.set_title(
    'softmax(new) / softmax(old)',
    color=color_palette["text"], fontsize=16, weight='bold'
)
ax2.plot(
    negative_logit_magnitudes, unstable_ratios,
    color=color_palette["accent"],
    linewidth=2
)
ax2.axhline(
    y=true_ratio,
    color=color_palette["secondary"],
    linewidth=1,
    linestyle='--'
)

failure_point_idx = np.where(unstable_ratios == 0)[0]
if failure_point_idx.size > 0:
    failure_point_x = negative_logit_magnitudes[failure_point_idx[0]]
    ax2.axvline(
        x=failure_point_x,
        color=color_palette["accent"],
        linestyle=':',
        linewidth=2,
        label=f'Underflow Failure at ~{failure_point_x:.0f}'
    )
    ax2.annotate(
        'Method fails here!\n(softmax underflows to 0)',
        xy=(failure_point_x, 0.2),
        xytext=(failure_point_x + 50, 0.6),
        arrowprops=dict(facecolor=color_palette["accent"], shrink=0.05, width=1, headwidth=8, edgecolor=color_palette["text"]),
        fontsize=10,
        color=color_palette["text"],
        bbox=dict(boxstyle="round,pad=0.3", fc=color_palette["background"], ec=color_palette["secondary"], lw=1)
    )

ax2.set_ylabel('Calculated Ratio', color=color_palette["text"], fontsize=12)
ax2.set_ylim(-0.1, true_ratio + 0.5)
ax2.tick_params(colors=color_palette["text"], labelsize=TICK_FONT_SIZE)
for spine in ax2.spines.values():
    spine.set_edgecolor(color_palette["secondary"])
ax2.set_xlabel('Magnitude of Inputs', color=color_palette["text"], fontsize=12)

# --- Final Touches ---
plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust rect to fit suptitle
plt.savefig(
    'ppo_stability_precision_chart.png',
    dpi=300,
    bbox_inches='tight',
    facecolor=fig.get_facecolor()
)

plt.close()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=FIG_SIZE, sharex=True)


ax1.plot(
    negative_logit_magnitudes, stable_error,
    label='Stable Method Error',
    color=color_palette["accent"], # Match color from plot 1
    linewidth=2
)
ax1.set_facecolor(color_palette["background"])
ax1.axhline(y=0, color=color_palette["secondary"], linewidth=1, linestyle='--')
ax1.set_ylabel('Calculation Error', color=color_palette["text"], fontsize=12)
ax1.set_ylim(-6e-14, 6e-14)
ax1.set_xlim(100, 700)
formatter = mticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-2, 2))
ax1.yaxis.set_major_formatter(formatter)
ax1.tick_params(colors=color_palette["text"], labelsize=TICK_FONT_SIZE)
for spine in ax1.spines.values():
    spine.set_edgecolor(color_palette["secondary"])
ax1.set_title(
    'exp(log_softmax(new) - log_softmax(old))',
    color=color_palette["text"], fontsize=16, weight='bold'
)


valid_unstable_idx = ~np.isnan(unstable_error) & (unstable_ratios != 0)
ax2.plot(
    negative_logit_magnitudes[valid_unstable_idx], unstable_error[valid_unstable_idx],
    label='Unstable Method Error',
    color=color_palette["accent"], # Match color from plot 2
    linewidth=1.5,
)
ax2.set_facecolor(color_palette["background"])
ax2.axhline(y=0, color=color_palette["secondary"], linewidth=1, linestyle='--')
ax2.set_xlabel('Magnitude of Inputs', color=color_palette["text"], fontsize=12)
ax2.set_ylabel('Calculation Error', color=color_palette["text"], fontsize=12)
ax2.set_ylim(-6e-14, 6e-14)
ax2.set_xlim(100, 700)
ax2.yaxis.set_major_formatter(formatter)
ax2.tick_params(colors=color_palette["text"], labelsize=TICK_FONT_SIZE)
for spine in ax2.spines.values():
    spine.set_edgecolor(color_palette["secondary"])
ax2.set_title(
    'softmax(new) / softmax(old)',
    color=color_palette["text"], fontsize=16, weight='bold'
)

plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust rect to fit suptitle
plt.savefig(
    'ppo_stability_precision_chart_zoom.png',
    dpi=300,
    bbox_inches='tight',
    facecolor=fig.get_facecolor()
)
