import pandas as pd
import glob
import os

csv_files = glob.glob("checkpoints/champions/trial_*_telemetry.csv")

def min_max_norm(series):
    return (series - series.min()) / (series.max() - series.min() + 1e-8)

for csv in csv_files:
    filename = os.path.basename(csv)
    trial_id = filename.split("_")[1]
    df = pd.read_csv(csv)
    
    df = df[df['epoch'] > 50].copy()
    
    norm_gp = min_max_norm(df['gp_nlpd'])
    norm_jepa = min_max_norm(df['jepa_loss'])
    norm_sr = 1.0 - min_max_norm(df['stable_rank'])
    
    df['score'] = (norm_gp + norm_jepa + norm_sr) / 3.0
    best = df.nsmallest(1, 'score').iloc[0]
    
    print(f"Trial {trial_id} | Best Epoch: {int(best['epoch'])}")
    print(f"  Score: {best['score']:.4f}")
    print(f"  GP NLPD: {best['gp_nlpd']:.2f}")
    print(f"  JEPA Loss: {best['jepa_loss']:.4f}")
    print(f"  Stable Rank: {best['stable_rank']:.4f}\n")

