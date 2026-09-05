import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

csv_files = glob.glob("checkpoints/champions/trial_*_telemetry.csv")
if not csv_files:
    print("No telemetry CSV files found. Wait for training to finish.")
    exit()

os.makedirs("graphs", exist_ok=True)

# 1. Loss Convergence Plot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for csv in csv_files:
    trial_id = csv.split("_")[2]
    df = pd.read_csv(csv)
    
    # Plot JEPA Loss
    axes[0].plot(df['epoch'], df['jepa_loss'], label=f"Trial {trial_id}")
    
    # Plot GP NLPD
    axes[1].plot(df['epoch'], df['gp_nlpd'], label=f"Trial {trial_id}")

axes[0].set_title("JEPA Loss Convergence (1000 Epochs)")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Validation JEPA Loss")
axes[0].set_yscale('log')
axes[0].legend()

axes[1].set_title("GP NLPD Convergence (1000 Epochs)")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Validation GP NLPD")
axes[1].legend()

plt.tight_layout()
plt.savefig("graphs/champions_loss_convergence.png", dpi=300)
plt.close()

# 2. JEPA Health (Stable Rank)
plt.figure(figsize=(10, 6))
for csv in csv_files:
    trial_id = csv.split("_")[2]
    df = pd.read_csv(csv)
    plt.plot(df['epoch'], df['stable_rank'], label=f"Trial {trial_id}")

plt.title("Latent Space Dimensionality (Stable Rank) over 1000 Epochs")
plt.xlabel("Epoch")
plt.ylabel("Stable Rank")
plt.legend()
plt.savefig("graphs/champions_stable_rank.png", dpi=300)
plt.close()

# 3. Individual Dimension Variance Tracker
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for i, csv in enumerate(csv_files):
    if i >= 4:
        break
    trial_id = csv.split("_")[2]
    df = pd.read_csv(csv)
    
    dim_cols = [c for c in df.columns if c.startswith("dim_")]
    for dim_col in dim_cols:
        axes[i].plot(df['epoch'], df[dim_col], alpha=0.7, label=dim_col)
        
    axes[i].set_title(f"Trial {trial_id} - Individual Dimension Variances")
    axes[i].set_xlabel("Epoch")
    axes[i].set_ylabel("Variance")

plt.tight_layout()
plt.savefig("graphs/champions_dim_variances.png", dpi=300)
plt.close()

print("Champion graphs generated successfully in graphs/")
