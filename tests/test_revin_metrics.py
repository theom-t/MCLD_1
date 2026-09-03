import torch
import torch.nn as nn
import polars as pl
import numpy as np

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine_weight = nn.Parameter(torch.ones(num_features))
        self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        return x

    def _get_statistics(self, x):
        self.mean = torch.mean(x, dim=1, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x):
        x = (x - self.affine_bias) / self.affine_weight
        x = x * self.stdev
        x = x + self.mean
        return x

def run_systematic_revin_audit():
    print("--- RevIN Systematic Evaluation Audit ---")
    
    # Load the raw PCHIP interpolated data (not the differenced data)
    df = pl.read_parquet("data/processed/monthly_interpolated.parquet").to_pandas()
    
    countries = df["country"].unique()
    features = df["feature"].unique()
    
    WINDOW_SIZE = 36
    revin = RevIN(num_features=1)
    
    total_windows = 0
    mean_list = []
    var_list = []
    corr_list = []
    mse_list = []
    
    print(f"Auditing {len(countries)} countries and {len(features)} features using {WINDOW_SIZE}-month rolling windows...")
    
    for feature in features:
        feature_data = df[df["feature"] == feature]
        
        for country in countries:
            c_f_data = feature_data[feature_data["country"] == country].sort_values("date")
            vals = c_f_data["value"].values
            
            # Skip if less than one window
            if len(vals) < WINDOW_SIZE:
                continue
                
            # Create rolling windows
            # To save time, we will sample non-overlapping windows
            for i in range(0, len(vals) - WINDOW_SIZE, WINDOW_SIZE):
                window = vals[i : i+WINDOW_SIZE]
                
                if np.isnan(window).any():
                    continue
                    
                total_windows += 1
                
                # Convert to [Batch=1, Time=36, Features=1]
                t_window = torch.tensor(window, dtype=torch.float64).unsqueeze(0).unsqueeze(-1)
                revin = revin.double()
                
                # 1. Forward Normalize
                t_norm = revin(t_window, mode='norm')
                
                # Metrics on Normalized Data
                norm_mean = t_norm.mean().item()
                norm_var = t_norm.var(unbiased=False).item()
                
                mean_list.append(norm_mean)
                var_list.append(norm_var)
                
                # 2. Signal Preservation (Pearson Correlation)
                raw_np = t_window.squeeze().detach().numpy()
                norm_np = t_norm.squeeze().detach().numpy()
                
                # If variance is zero (flat line), corr is undefined, skip correlation
                if np.std(raw_np) > 1e-6 and np.std(norm_np) > 1e-6:
                    corr = np.corrcoef(raw_np, norm_np)[0, 1]
                    if not np.isnan(corr):
                        corr_list.append(corr)
                
                # 3. Reverse Denormalize
                t_recon = revin(t_norm, mode='denorm')
                recon_np = t_recon.squeeze().detach().numpy()
                
                # Reconstruction Error (MSE)
                mse = np.mean((raw_np - recon_np)**2)
                mse_list.append(mse)
                
    print("\n--- Formal RevIN Spec Metrics ---")
    print(f"Total Windows Evaluated: {total_windows:,}")
    
    print("\n1. Internal Stationarity (Target: Mean=0.0, Var=1.0)")
    avg_mean = np.nanmean(mean_list)
    avg_var = np.nanmean(var_list)
    print(f"   -> Average Normalized Mean: {avg_mean:.6f}")
    print(f"   -> Average Normalized Variance: {avg_var:.6f}")
    
    print("\n2. Signal Preservation (Target: Correlation = 1.0)")
    avg_corr = np.nanmean(corr_list)
    print(f"   -> Average Pearson Correlation (Raw vs Normalized): {avg_corr:.6f}")
    
    print("\n3. Lossless Reconstructability (Target: MSE < 1e-5)")
    avg_mse = np.nanmean(mse_list)
    print(f"   -> Average Reconstruction MSE: {avg_mse:.10e}")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    run_systematic_revin_audit()
