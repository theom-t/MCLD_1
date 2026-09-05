"""
MCLD-1 Joint Training Demo

Runs a short training loop on the REAL macro data to verify
the full pipeline from Data Bridge → JEPA → SVGP is functional.
"""
import jax
import jax.numpy as jnp
import optax
import numpy as np
import time
from tqdm import tqdm

from src.utils.config import JepaConfig
from src.models.jepa.model import JEPA, Predictor
from src.models.gp.model import build_svgp
from src.pipeline.train_step import create_joint_train_step, create_eval_step
from src.data.window_generator import MacroDataBridge


def run_demo():
    print("=" * 70)
    print("🚀 MCLD-1 Joint Training Demo (Real Data)")
    print("=" * 70)

    # 1. Load Real Data
    config = JepaConfig(
        context_window_months=36,
        target_horizon_months=6,
        latent_dim=6,
        encoder_hidden_dim=64,
        encoder_layers=3,
        gp_num_inducing=128,  # Smaller for fast demo
        gp_kernel_type="Matern52",
        gp_periodic_active=True,
        gp_learning_rate=0.01,
        learning_rate=1e-3,
        batch_size=128,
    )

    bridge = MacroDataBridge(config)
    bridge.build_datasets()

    # Update num_features from actual data
    actual_features = bridge._actual_num_features
    config = JepaConfig(
        context_window_months=36,
        target_horizon_months=6,
        num_features=actual_features,
        latent_dim=6,
        encoder_hidden_dim=64,
        encoder_layers=3,
        gp_num_inducing=128,
        gp_kernel_type="Matern52",
        gp_periodic_active=True,
        gp_learning_rate=0.01,
        learning_rate=1e-3,
        batch_size=128,
    )

    # 2. Initialize Models
    print("\nBuilding JEPA (Flax)...")
    key = jax.random.PRNGKey(42)
    key, enc_key, pred_key = jax.random.split(key, 3)

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

    jepa_optimizer = optax.adamw(learning_rate=config.learning_rate, weight_decay=config.weight_decay)
    jepa_state = {
        "encoder_params": encoder_params,
        "predictor_params": predictor_params,
        "target_encoder_params": target_params,
        "opt_state": {
            "encoder": jepa_optimizer.init(encoder_params),
            "predictor": jepa_optimizer.init(predictor_params),
        },
    }

    # GP Model
    print("Building SVGP (GPJax)...")
    train_n = bridge.datasets["train"]["context_window"].shape[0]
    svgp_q = build_svgp(config, train_n=train_n)
    gp_optimizer = optax.adam(learning_rate=config.gp_learning_rate)
    gp_opt_state = gp_optimizer.init(svgp_q)

    # Compile
    print("JIT Compiling Joint Step (Flax + GPJax + StopGradient Firewall)...")
    t0 = time.time()
    train_step = create_joint_train_step(encoder, predictor, jepa_optimizer, config)
    eval_step = create_eval_step(encoder, predictor, config)

    # 3. Training Loop
    num_epochs = 50
    print(f"\nStarting Training Loop ({num_epochs} epochs)...\n")
    
    loop_key = jax.random.PRNGKey(1337)

    for epoch in range(num_epochs):
        # --- TRAIN ---
        train_gen = bridge.get_batch_generator("train", config.batch_size, shuffle=True, seed=epoch)
        train_metrics_list = []

        for batch in train_gen:
            loop_key, step_key = jax.random.split(loop_key)
            jepa_state, gp_opt_state, svgp_q, metrics = train_step(
                jepa_state, gp_opt_state, gp_optimizer, svgp_q, batch, step_key
            )
            m = jax.tree_util.tree_map(lambda x: float(x), metrics)
            train_metrics_list.append(m)

        if not train_metrics_list:
            print(f"Epoch {epoch+1}: No training batches!")
            continue

        avg_train = {k: np.nanmean([m[k] for m in train_metrics_list]) for k in train_metrics_list[0]}

        # --- EVAL ---
        val_gen = bridge.get_batch_generator("val", config.batch_size)
        val_metrics_list = []

        for batch in val_gen:
            vm = eval_step(jepa_state, svgp_q, batch)
            vm = jax.tree_util.tree_map(lambda x: float(x), vm)
            val_metrics_list.append(vm)

        if val_metrics_list:
            avg_val = {k: np.nanmean([m[k] for m in val_metrics_list]) for k in val_metrics_list[0]}
        else:
            avg_val = {}

        # Print
        val_sr = avg_val.get("eval_stable_rank", 0)
        val_gp = avg_val.get("eval_gp_nelbo", float("nan"))
        print(
            f"Epoch {epoch+1:02d} | "
            f"Train JEPA: {avg_train['loss_jepa']:>8.2f} | "
            f"Train GP: {avg_train['loss_gp_nelbo']:>10.2f} | "
            f"Train SR: {avg_train['stable_rank']:.2f} | "
            f"Val GP: {val_gp:>10.2f} | "
            f"Val SR: {val_sr:.2f}"
        )

    elapsed = time.time() - t0
    print(f"\n✅ Demo Complete! {num_epochs} epochs in {elapsed:.1f}s")


if __name__ == "__main__":
    run_demo()
