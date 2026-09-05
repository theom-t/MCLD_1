import jax
import jax.numpy as jnp
import optax
import optuna
import os
import pandas as pd
import time
import equinox as eqx

from src.utils.config import JepaConfig
from src.data.window_generator import MacroDataBridge
from src.models.jepa.model import JEPA, Predictor
from src.models.gp.model import build_svgp
from src.pipeline.train_step import create_joint_train_step, create_eval_step

def get_champion_params(trial_id: int):
    study = optuna.load_study(study_name="MCLD-1-Joint-Architecture", storage="sqlite:///mcld1_optuna.db")
    for trial in study.trials:
        if trial.number == trial_id:
            return trial.params
    raise ValueError(f"Trial {trial_id} not found in DB.")

def train_champion(trial_id: int, max_epochs: int = 1000):
    print(f"\n{'='*60}")
    print(f"🏆 TRAINING CHAMPION MODEL: Trial {trial_id}")
    print(f"{'='*60}")
    
    # 1. Extract hyperparameters
    params = get_champion_params(trial_id)
    config = JepaConfig(
        encoder_layers=params["encoder_layers"],
        encoder_hidden_dim=params["encoder_hidden"],
        latent_dim=params["latent_dim"],
        loss_var_base_weight=params["var_weight"],
        loss_cov_weight=params["cov_weight"],
        loss_topo_weight=params["topo_weight"],
        ema_tau_base=params["ema_tau"],
        aug_gaussian_noise=params["aug_noise"],
        aug_feature_dropout=params["aug_dropout"],
        gp_num_inducing=params["gp_num_inducing"],
        gp_kernel_type=params["gp_kernel"],
        gp_periodic_active=params["gp_periodic"],
        gp_kernel_composition=params["gp_composition"],
        gp_mean_type=params["gp_mean"]
    )
    # Lock batch size to 512 for stability
    config.batch_size = 512

    # 2. Load Data
    data_bridge = MacroDataBridge(config, "data/processed")
    data_bridge.build_datasets()
    
    # 3. Initialize Models
    key = jax.random.PRNGKey(42 + trial_id)
    key, init_key, gp_key, train_key = jax.random.split(key, 4)
    enc_key, pred_key = jax.random.split(init_key)
    
    input_dim = data_bridge.datasets["train"]["context_window"].shape[-1]
    config.num_features = input_dim
    
    encoder = JEPA(
        num_features=config.num_features,
        hidden_dim=config.encoder_hidden_dim,
        latent_dim=config.latent_dim,
        num_layers=config.encoder_layers,
        kernel_size=config.encoder_kernel_size,
    )
    predictor = Predictor(
        hidden_dim=config.predictor_hidden_dim,
        latent_dim=config.latent_dim,
        num_layers=config.predictor_layers,
    )
    
    dummy_x = jnp.zeros((config.batch_size, config.context_window_months, config.num_features))
    dummy_m = jnp.ones_like(dummy_x)

    encoder_params = encoder.init(enc_key, dummy_x, dummy_m)
    target_params = encoder.init(enc_key, dummy_x, dummy_m)
    predictor_params = predictor.init(pred_key, jnp.zeros((config.batch_size, config.latent_dim)))
    
    train_n = data_bridge.datasets["train"]["context_window"].shape[0]
    svgp_q = build_svgp(config, train_n=train_n)
    
    # 4. Optimizers
    jepa_optimizer = optax.adamw(learning_rate=params["lr"], weight_decay=1e-4)
    gp_optimizer = optax.adam(learning_rate=params["gp_lr"])
    
    opt_state_jepa = {
        "encoder": jepa_optimizer.init(encoder_params),
        "predictor": jepa_optimizer.init(predictor_params),
    }
    
    jepa_state = {
        "encoder_params": encoder_params,
        "predictor_params": predictor_params,
        "target_encoder_params": target_params,
        "opt_state": opt_state_jepa,
    }
    
    gp_opt_state = gp_optimizer.init(eqx.filter(svgp_q, eqx.is_inexact_array))
    
    # 5. JIT Compiling Steps
    train_step = create_joint_train_step(encoder, predictor, jepa_optimizer, config)
    eval_step = create_eval_step(encoder, predictor, config)
    
    # 6. Training Loop with Telemetry
    telemetry = []
    
    for epoch in range(max_epochs):
        train_gen = data_bridge.get_batch_generator("train", config.batch_size, shuffle=True)
        
        for batch in train_gen:
            train_key, step_key = jax.random.split(train_key)
            jepa_state, gp_opt_state, svgp_q, _ = train_step(
                jepa_state, gp_opt_state, gp_optimizer, svgp_q, batch, step_key
            )
            
        # Eval Pass
        val_gen = data_bridge.get_batch_generator("val", config.batch_size, shuffle=False)
        val_metrics_list = []
        for batch in val_gen:
            metrics = eval_step(jepa_state, svgp_q, batch)
            m = jax.tree_util.tree_map(lambda x: float(x) if x.ndim == 0 else x, metrics)
            val_metrics_list.append(m)
            
        avg = {}
        for k in val_metrics_list[0].keys():
            avg[k] = sum(m[k] for m in val_metrics_list) / len(val_metrics_list)
            
        # Parse dimensions
        dim_vars = avg["eval_var_dims"]
        dim_dict = {f"dim_{i}_var": float(dim_vars[i]) for i in range(len(dim_vars))}
        
        row = {
            "epoch": epoch,
            "gp_nlpd": float(avg["eval_gp_nelbo"]),
            "jepa_loss": float(avg["eval_topo"] + avg["eval_var_hinge"] + avg["eval_cov"]),
            "stable_rank": float(avg["eval_stable_rank"]),
            "var_hinge": float(avg["eval_var_hinge"]),
            "cov_loss": float(avg["eval_cov"]),
            "topo_loss": float(avg["eval_topo"]),
            **dim_dict
        }
        telemetry.append(row)
        
        if epoch % 50 == 0 or epoch == max_epochs - 1:
            print(f"Epoch {epoch:04d} | GP NLPD: {row['gp_nlpd']:>9.2f} | JEPA: {row['jepa_loss']:>6.2f} | SR: {row['stable_rank']:.2f}")

    # --- SAVE ARTIFACTS ---
    os.makedirs("checkpoints/champions", exist_ok=True)
    
    # Save Telemetry
    df = pd.DataFrame(telemetry)
    csv_path = f"checkpoints/champions/trial_{trial_id}_telemetry.csv"
    df.to_csv(csv_path, index=False)

    # Save Weights
    eqx.tree_serialise_leaves(f"checkpoints/champions/trial_{trial_id}_encoder.eqx", jepa_state["encoder_params"])
    eqx.tree_serialise_leaves(f"checkpoints/champions/trial_{trial_id}_predictor.eqx", jepa_state["predictor_params"])
    
    # Clean up XLA Cache
    jax.clear_caches()
    import gc
    gc.collect()
    
    print(f"Saved telemetry to {csv_path}")

if __name__ == "__main__":
    champions = [275, 250, 13, 14]
    for c in champions:
        train_champion(c, max_epochs=1000)
        
