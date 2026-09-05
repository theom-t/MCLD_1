import optuna
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Load study
study = optuna.load_study(study_name="MCLD-1-Joint-Architecture", storage="sqlite:///mcld1_optuna.db")
completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

if len(completed_trials) == 0:
    print("No completed trials.")
    exit()

data = []
for t in completed_trials:
    gp_nlpd = t.values[0]
    jepa_loss = t.values[1]
    sr = -t.values[2]  # Negative SR (we want to minimize this)
    data.append({
        "trial_id": t.number,
        "gp_nlpd": gp_nlpd,
        "jepa_loss": jepa_loss,
        "neg_sr": sr,
        **t.params
    })

df = pd.DataFrame(data)

# Normalize the 3 objectives to [0, 1] (Lower is better for all 3 now)
def min_max_norm(series):
    return (series - series.min()) / (series.max() - series.min() + 1e-8)

df['norm_gp'] = min_max_norm(df['gp_nlpd'])
df['norm_jepa'] = min_max_norm(df['jepa_loss'])
# Invert SR normalization so that a HIGHER Stable Rank maps to a LOWER score (penalty)
df['norm_sr'] = 1.0 - min_max_norm(df['neg_sr'])

# Combined score (Lower is better)
df['combined_score'] = (df['norm_gp'] + df['norm_jepa'] + df['norm_sr']) / 3.0

print(f"Computed combined scores for {len(df)} trials.")
print("Top 5 Combined Configurations (Lowest is Best):")
print(df.nsmallest(5, 'combined_score')[['trial_id', 'combined_score', 'gp_nlpd', 'jepa_loss', 'neg_sr']])

# List of numeric parameters
params = [col for col in df.columns if col not in 
          ['trial_id', 'gp_nlpd', 'jepa_loss', 'neg_sr', 'norm_gp', 'norm_jepa', 'norm_sr', 'combined_score', 'gp_kernel', 'gp_composition', 'gp_mean']]

# Generate Slice Plots
num_params = len(params)
cols = 3
rows = int(np.ceil(num_params / cols))

fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
axes = axes.flatten()

for i, param in enumerate(params):
    ax = axes[i]
    ax.scatter(df[param], df['combined_score'], c=df['combined_score'], cmap='viridis_r', alpha=0.7)
    
    # If the param has a very wide range, log scale might be better, but Optuna handles that. 
    # We just plot the raw values.
    if param in ['lr', 'gp_lr']:
        ax.set_xscale('log')
        
    ax.set_xlabel(param)
    ax.set_ylabel('Combined Score (Lower is Better)')
    ax.set_title(f'{param} vs Combined Score')

# Hide empty subplots
for i in range(num_params, len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
os.makedirs('graphs', exist_ok=True)
plt.savefig('graphs/slice_plots.png', dpi=300)
print("\nSaved slice plots to graphs/slice_plots.png")

# Also print correlations to help analyze what is limiting development
print("\nParameter Correlations with Combined Score (Negative means higher param = better score):")
corr = df[params].corrwith(df['combined_score']).sort_values()
print(corr)

