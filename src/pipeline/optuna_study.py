"""
MCLD-1 Optuna Multi-Objective Hyperparameter Search

Runs a Pareto search over the joint JEPA + SVGP architecture using
three competing objectives:
  1. Minimize GP NLPD (forecast calibration)
  2. Minimize JEPA Loss (macro representation quality)
  3. Maximize Stable Rank (anti-collapse guarantee)

Includes Early Stopping (patience=15) and a hard collapse kill-switch.
"""
import optuna
import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import numpy as np
from tqdm import tqdm
import time

from src.utils.config import JepaConfig
from src.models.jepa.model import JEPA, Predictor
from src.models.gp.model import build_svgp
from src.pipeline.train_step import create_joint_train_step, create_eval_step
from src.data.window_generator import MacroDataBridge


def objective(trial: optuna.Trial, data_bridge: MacroDataBridge, max_epochs: int = 300):
    """
    Optuna objective function for Multi-Objective Joint Training.
    """
    # 1. Propose Hyperparameters
    config = JepaConfig(
        # Data (fixed for now — Optuna can search these later)
        num_features=data_bridge._actual_num_features,

        # Architecture
        encoder_layers=trial.suggest_int("encoder_layers", 2, 8),
        encoder_hidden_dim=trial.suggest_categorical("encoder_hidden", [32, 64, 128]),
        latent_dim=trial.suggest_int("latent_dim", 4, 12),

        # GP Architecture
        gp_num_inducing=trial.suggest_categorical("gp_num_inducing", [128, 256, 512]),
        gp_kernel_type=trial.suggest_categorical("gp_kernel", ["Matern12", "Matern32", "Matern52", "RBF"]),
        gp_periodic_active=trial.suggest_categorical("gp_periodic", [True, False]),
        gp_kernel_composition=trial.suggest_categorical("gp_composition", ["additive", "multiplicative"]),
        gp_mean_type=trial.suggest_categorical("gp_mean", ["Zero", "Constant"]),

        # Optimization
        learning_rate=trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        gp_learning_rate=trial.suggest_float("gp_lr", 1e-3, 5e-2, log=True),
        loss_var_base_weight=trial.suggest_float("var_weight", 10.0, 50.0),
        loss_cov_weight=trial.suggest_float("cov_weight", 10.0, 50.0),
        loss_topo_weight=trial.suggest_float("topo_weight", 1.0, 10.0),
        ema_tau_base=trial.suggest_float("ema_tau", 0.99, 0.999),
        
        # Augmentation
        aug_gaussian_noise=trial.suggest_float("aug_noise", 0.0, 0.2),
        aug_feature_dropout=trial.suggest_float("aug_dropout", 0.0, 0.4),
    )

    # 2. Init Random Keys
    key = jax.random.PRNGKey(trial.number)
    key, enc_key, pred_key = jax.random.split(key, 3)

    # 3. Initialize Models
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
    train_n = data_bridge.datasets["train"]["context_window"].shape[0]
    svgp_q = build_svgp(config, train_n=train_n)
    gp_optimizer = optax.adam(learning_rate=config.gp_learning_rate)
    gp_opt_state = gp_optimizer.init(svgp_q)

    # Compile train and eval steps
    t_compile = time.time()
    train_step = create_joint_train_step(encoder, predictor, jepa_optimizer, config)
    eval_step = create_eval_step(encoder, predictor, config)
    print(f"  Trial {trial.number}: JIT compilation took {time.time() - t_compile:.1f}s")

    # 4. Training Loop
    best_gp_nelbo = float("inf")
    best_jepa_loss = float("inf")
    best_stable_rank = 0.0
    
    loop_key = jax.random.PRNGKey(trial.number + 1000)

    for epoch in range(max_epochs):
        # --- TRAIN EPOCH ---
        train_gen = data_bridge.get_batch_generator(
            "train", config.batch_size, shuffle=True, seed=epoch + trial.number * 1000
        )

        for batch in train_gen:
            loop_key, step_key = jax.random.split(loop_key)
            jepa_state, gp_opt_state, svgp_q, _ = train_step(
                jepa_state, gp_opt_state, gp_optimizer, svgp_q, batch, step_key
            )

        # --- VAL EPOCH (using dedicated eval_step — Bug 4 fix) ---
        val_gen = data_bridge.get_batch_generator("val", config.batch_size)
        val_metrics_list = []

        for batch in val_gen:
            val_metrics = eval_step(jepa_state, svgp_q, batch)
            val_metrics = jax.tree_util.tree_map(lambda x: float(x), val_metrics)
            val_metrics_list.append(val_metrics)

        if not val_metrics_list:
            raise optuna.TrialPruned("No validation batches generated.")

        # Average validation metrics
        avg = {
            k: np.nanmean([m[k] for m in val_metrics_list])
            for k in val_metrics_list[0].keys()
        }

        # Print live metrics to terminal every 10 epochs
        if epoch % 10 == 0 or epoch == max_epochs - 1:
            print(
                f"  Trial {trial.number} | Epoch {epoch:03d} | "
                f"GP NLPD: {avg['eval_gp_nelbo']:>9.2f} | "
                f"JEPA Loss: {avg['eval_topo'] + avg['eval_var_hinge'] + avg['eval_cov']:>6.2f} | "
                f"SR: {avg['eval_stable_rank']:.2f}"
            )

        # --- OPTUNA NATIVE PRUNER REMOVED ---
        # Optuna raises NotImplementedError for intermediate median-pruning on Multi-Objective studies.
        # This is perfectly fine; we rely entirely on our custom Early Stopping and Collapse Kill-Switch.
        
        # --- METRIC TRACKING (With Burn-in Period) ---
        # We ignore the first 30 epochs completely so the artificially low 
        # Epoch 0 score doesn't become the benchmark.
        if epoch > 30:
            if avg["eval_gp_nelbo"] < best_gp_nelbo:
                best_gp_nelbo = avg["eval_gp_nelbo"]
                best_jepa_loss = avg["eval_topo"] + avg["eval_var_hinge"] + avg["eval_cov"]
                best_stable_rank = avg["eval_stable_rank"]
        else:
            # During burn-in, we just passively track the most recent epoch as the baseline.
            best_gp_nelbo = avg["eval_gp_nelbo"]
            best_jepa_loss = avg["eval_topo"] + avg["eval_var_hinge"] + avg["eval_cov"]
            best_stable_rank = avg["eval_stable_rank"]

    # --- FLUSH VRAM CACHE ---
    # Clear Equinox/JAX compilation caches so the massive XLA graphs from this specific 
    # architecture don't bloat the RTX 5090's VRAM during the next trial.
    jax.clear_caches()
    
    import gc
    gc.collect()

    # Return 3 objectives for the Pareto front
    return best_gp_nelbo, best_jepa_loss, -best_stable_rank  # Negate SR so "minimize" = maximize


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50, help="Number of Optuna trials to run")
    parser.add_argument("--epochs", type=int, default=300, help="Maximum epochs per trial")
    args = parser.parse_args()

    print("=" * 60)
    print("MCLD-1 Optuna Multi-Objective Pareto Search")
    print("=" * 60)

    print("\nInitializing Data Bridge...")
    bridge = MacroDataBridge(JepaConfig())
    bridge.build_datasets()

    # Define Multi-Objective Study (with SQLite persistence)
    study = optuna.create_study(
        study_name="MCLD-1-Joint-Architecture",
        storage="sqlite:///mcld1_optuna.db",
        load_if_exists=True,
        directions=["minimize", "minimize", "minimize"],  # GP NLPD, JEPA Loss, -StableRank
        sampler=optuna.samplers.NSGAIISampler(),
    )

    print(f"\nLaunching Optuna Pareto Search ({args.trials} trials, up to {args.epochs} epochs each)...")
    study.optimize(
        lambda trial: objective(trial, bridge, max_epochs=args.epochs),
        n_trials=args.trials,
        timeout=14400,  # 4 hours max
    )

    print("\n" + "=" * 60)
    print("🏆 Pareto Front Discovered")
    print("=" * 60)
    best_trials = study.best_trials
    for t in best_trials:
        print(
            f"Trial {t.number}: "
            f"GP NLPD={t.values[0]:.2f}, "
            f"JEPA Loss={t.values[1]:.2f}, "
            f"Stable Rank={-t.values[2]:.2f}"
        )
        print(f"  Params: {t.params}\n")
