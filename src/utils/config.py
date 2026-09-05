from dataclasses import dataclass
from typing import Tuple

@dataclass
class JepaConfig:
    # --- Data & Temporal Parameters ---
    context_window_months: int = 36
    target_horizon_months: int = 6
    target_gap_months: int = 0
    num_features: int = 23
    
    # --- Architecture Parameters ---
    latent_dim: int = 6
    encoder_hidden_dim: int = 64
    encoder_layers: int = 3
    encoder_kernel_size: int = 3
    predictor_hidden_dim: int = 32
    predictor_layers: int = 2
    
    # --- Loss & Training Hyperparameters ---
    learning_rate: float = 0.000868
    weight_decay: float = 1e-4
    ema_tau_base: float = 0.9976
    loss_inv_weight: float = 25.0
    loss_var_base_weight: float = 46.65
    loss_cov_weight: float = 45.74
    loss_topo_weight: float = 9.77
    
    # Data Augmentation
    aug_gaussian_noise: float = 0.109
    aug_feature_dropout: float = 0.378
    batch_size: int = 512
    
    # --- GP / Stage 3 Parameters ---
    gp_num_inducing: int = 512
    gp_kernel_type: str = "Matern32"
    gp_mean_type: str = "Constant"
    gp_kernel_composition: str = "additive"
    gp_learning_rate: float = 0.001477
    gp_periodic_active: bool = False
