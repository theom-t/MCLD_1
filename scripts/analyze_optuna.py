import optuna
import matplotlib.pyplot as plt
import pandas as pd

# Load study
study = optuna.load_study(study_name="MCLD-1-Joint-Architecture", storage="sqlite:///mcld1_optuna.db")

print(f"Total trials in DB: {len(study.trials)}")

# Filter completed trials
completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
print(f"Completed trials: {len(completed_trials)}")

if len(completed_trials) == 0:
    print("No completed trials to plot.")
    exit()

data = []
for t in completed_trials:
    gp_nlpd = t.values[0]
    jepa_loss = t.values[1]
    sr = -t.values[2]  # Invert back to positive
    data.append({
        "trial_id": t.number,
        "gp_nlpd": gp_nlpd,
        "jepa_loss": jepa_loss,
        "stable_rank": sr,
        **t.params
    })

df = pd.DataFrame(data)

# Print best configs
print("\nTop 5 by GP NLPD (Lowest):")
print(df.nsmallest(5, 'gp_nlpd')[['trial_id', 'gp_nlpd', 'jepa_loss', 'stable_rank']])

print("\nTop 5 by JEPA Loss (Lowest):")
print(df.nsmallest(5, 'jepa_loss')[['trial_id', 'gp_nlpd', 'jepa_loss', 'stable_rank']])

print("\nTop 5 by Stable Rank (Highest):")
print(df.nlargest(5, 'stable_rank')[['trial_id', 'gp_nlpd', 'jepa_loss', 'stable_rank']])

# Plotting 3D Pareto Front
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(df['gp_nlpd'], df['jepa_loss'], df['stable_rank'], c=df['stable_rank'], cmap='viridis', s=50, alpha=0.8)
ax.set_xlabel('GP NLPD (Lower is Better)')
ax.set_ylabel('JEPA Loss (Lower is Better)')
ax.set_zlabel('Stable Rank (Higher is Better)')
ax.set_title(f'Optuna Pareto Front Search (Trials: {len(completed_trials)})')
plt.colorbar(sc, label='Stable Rank')
plt.savefig('graphs/pareto_front_3d.png', dpi=300, bbox_inches='tight')

# Plotting 2D Pareto Fronts (Pairwise)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].scatter(df['gp_nlpd'], df['jepa_loss'], c=df['stable_rank'], cmap='viridis', alpha=0.7)
axes[0].set_xlabel('GP NLPD')
axes[0].set_ylabel('JEPA Loss')
axes[0].set_title('GP NLPD vs JEPA Loss')

axes[1].scatter(df['gp_nlpd'], df['stable_rank'], c=df['stable_rank'], cmap='viridis', alpha=0.7)
axes[1].set_xlabel('GP NLPD')
axes[1].set_ylabel('Stable Rank')
axes[1].set_title('GP NLPD vs Stable Rank')

axes[2].scatter(df['jepa_loss'], df['stable_rank'], c=df['stable_rank'], cmap='viridis', alpha=0.7)
axes[2].set_xlabel('JEPA Loss')
axes[2].set_ylabel('Stable Rank')
axes[2].set_title('JEPA Loss vs Stable Rank')

plt.tight_layout()
plt.savefig('graphs/pareto_front_2d.png', dpi=300)
print("\nSaved Pareto front plots to graphs/")
