"""
MCLD-1 Joint Training Step: JEPA + SVGP

Implements the Decoupled Joint-Training architecture with a jax.lax.stop_gradient
firewall between the Temporal-JEPA and the Sparse Variational GP.

The JEPA is updated exclusively by its self-supervised losses (VICReg + Topological).
The SVGP is updated by its ELBO loss, but cannot influence the JEPA's weights.
"""
import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import gpjax as gpx
from typing import Dict, Any, Tuple

from src.models.jepa.step import vicreg_loss, macro_topological_loss
from src.utils.config import JepaConfig


def create_joint_train_step(
    encoder_model, predictor_model, jepa_optimizer, config: JepaConfig
):
    """
    Factory function that returns a JIT-compiled joint training step.
    """

    def loss_fn(
        jepa_params, predictor_params, target_encoder_params, svgp_q,
        batch: Dict[str, jnp.ndarray], step_key: jax.Array
    ):
        """Computes the full joint loss with the stop_gradient firewall."""
        # --- 1. JEPA FORWARD PASS ---
        context = batch["context_window"]       # (B, T_ctx, F)
        context_mask = batch["context_mask"]    # (B, T_ctx, F)
        target = batch["target_window"]         # (B, T_tgt, F)
        target_mask = batch["target_mask"]      # (B, T_tgt, F)
        
        # --- DATA AUGMENTATION (Applied to Context Only) ---
        if config.aug_gaussian_noise > 0.0 or config.aug_feature_dropout > 0.0:
            noise_key, drop_key = jax.random.split(step_key)
            
            if config.aug_gaussian_noise > 0.0:
                noise = jax.random.normal(noise_key, context.shape) * config.aug_gaussian_noise
                # Apply noise only to valid data points
                context = context + (noise * context_mask)
                
            if config.aug_feature_dropout > 0.0:
                keep_prob = 1.0 - config.aug_feature_dropout
                drop_mask = jax.random.bernoulli(drop_key, p=keep_prob, shape=context.shape).astype(jnp.float32)
                context = context * drop_mask
                context_mask = context_mask * drop_mask

        s_x, _, _ = encoder_model.apply(jepa_params, context, context_mask)   # (B, D)
        hat_s_y = predictor_model.apply(predictor_params, s_x)                # (B, D)

        s_y, _, _ = encoder_model.apply(target_encoder_params, target, target_mask)  # (B, D)
        s_y = jax.lax.stop_gradient(s_y)

        # --- 2. JEPA LOSSES ---
        # Cosine invariance loss
        pred_norm = hat_s_y / jnp.maximum(jnp.linalg.norm(hat_s_y, axis=-1, keepdims=True), 1e-12)
        targ_norm = s_y / jnp.maximum(jnp.linalg.norm(s_y, axis=-1, keepdims=True), 1e-12)
        loss_inv = jnp.mean(2.0 - 2.0 * jnp.sum(pred_norm * targ_norm, axis=-1))

        # Adaptive VICReg (PID-controlled variance + covariance)
        loss_var, loss_cov, batch_var = vicreg_loss(
            s_x, s_y,
            base_var_weight=config.loss_var_base_weight,
            cov_weight=config.loss_cov_weight,
        )

        # Macro-Topological Loss
        loss_topo = macro_topological_loss(context, s_x)

        loss_jepa = (
            (config.loss_inv_weight * loss_inv)
            + loss_var
            + loss_cov
            + (config.loss_topo_weight * loss_topo)
        )

        # --- 3. THE FIREWALL ---
        s_x_frozen = jax.lax.stop_gradient(s_x).astype(jnp.float64)  # (B, D)
        s_y_frozen = jax.lax.stop_gradient(s_y).astype(jnp.float64)  # (B, D)

        # --- 4. SVGP ELBO PASS ---
        batch_size = s_x_frozen.shape[0]
        t_context = jnp.zeros((batch_size, 1), dtype=jnp.float64)  # (B, 1)
        
        # GP input: concatenate latent state with temporal index -> (B, D+1)
        gp_x = jnp.concatenate([s_x_frozen, t_context], axis=-1)  # (B, D+1)
        gp_y = s_y_frozen  # (B, D) — multi-output targets

        gp_dataset = gpx.Dataset(X=gp_x, y=gp_y)
        elbo = gpx.objectives.elbo(svgp_q, gp_dataset)
        loss_gp = -elbo  # Minimize negative ELBO

        # --- 5. TOTAL JOINT LOSS ---
        total_loss = loss_jepa + loss_gp

        # --- 6. STABLE RANK (for Optuna kill-switch) ---
        s_vals = jnp.linalg.svd(s_x - jnp.mean(s_x, axis=0), compute_uv=False)
        stable_rank = jnp.sum(s_vals ** 2) / (jnp.max(s_vals) ** 2 + 1e-12)

        metrics = {
            "loss_total": total_loss,
            "loss_jepa": loss_jepa,
            "loss_gp_nelbo": loss_gp,
            "jepa_inv": loss_inv,
            "jepa_var": loss_var,
            "jepa_cov": loss_cov,
            "jepa_topo": loss_topo,
            "batch_variance": batch_var,
            "stable_rank": stable_rank,
        }

        return total_loss, metrics

    # Wrap so we can differentiate w.r.t a single "trainable" tuple
    def wrapped_loss_fn(trainable_params, target_encoder_params, batch, step_key):
        j_params, p_params, q_params = trainable_params
        return loss_fn(j_params, p_params, target_encoder_params, q_params, batch, step_key)

    @eqx.filter_jit
    def joint_train_step(
        jepa_state: Dict[str, Any],
        gp_opt_state,
        gp_optimizer,
        svgp_q,
        batch: Dict[str, jnp.ndarray],
        step_key: jax.Array
    ):
        """Single JIT-compiled joint training step."""
        jepa_params = jepa_state["encoder_params"]
        predictor_params = jepa_state["predictor_params"]
        target_encoder_params = jepa_state["target_encoder_params"]
        opt_state_jepa = jepa_state["opt_state"]

        trainable = (jepa_params, predictor_params, svgp_q)

        grad_fn = eqx.filter_value_and_grad(wrapped_loss_fn, has_aux=True)
        (loss, metrics), grads = grad_fn(trainable, target_encoder_params, batch, step_key)

        jepa_grads, predictor_grads, svgp_grads = grads

        # Update JEPA encoder
        enc_updates, new_opt_state_enc = jepa_optimizer.update(
            jepa_grads, opt_state_jepa["encoder"], jepa_params
        )
        new_jepa_params = optax.apply_updates(jepa_params, enc_updates)

        # Update Predictor
        pred_updates, new_opt_state_pred = jepa_optimizer.update(
            predictor_grads, opt_state_jepa["predictor"], predictor_params
        )
        new_predictor_params = optax.apply_updates(predictor_params, pred_updates)

        # EMA update for Target Encoder
        tau = config.ema_tau_base
        new_target_params = jax.tree_util.tree_map(
            lambda targ, ctx: tau * targ + (1.0 - tau) * ctx,
            target_encoder_params, new_jepa_params,
        )

        new_jepa_state = {
            "encoder_params": new_jepa_params,
            "predictor_params": new_predictor_params,
            "target_encoder_params": new_target_params,
            "opt_state": {
                "encoder": new_opt_state_enc,
                "predictor": new_opt_state_pred,
            },
        }

        # Update SVGP via Equinox/Optax
        gp_updates, new_gp_opt_state = gp_optimizer.update(
            svgp_grads, gp_opt_state, svgp_q
        )
        new_svgp_q = eqx.apply_updates(svgp_q, gp_updates)

        return new_jepa_state, new_gp_opt_state, new_svgp_q, metrics

    return joint_train_step


def create_eval_step(
    encoder_model, predictor_model, config: JepaConfig
):
    """
    Factory function that returns a JIT-compiled evaluation step.

    BUG 4 FIX: This is a pure evaluation function that does NOT use the PID-adaptive
    variance weighting. It computes raw, comparable metrics across epochs.

    Returns:
        A JIT-compiled function: (jepa_state, svgp_q, batch) -> metrics
    """

    def eval_loss_fn(
        jepa_params, target_encoder_params, svgp_q,
        batch: Dict[str, jnp.ndarray],
    ):
        """Computes evaluation metrics without adaptive weighting."""
        context = batch["context_window"]
        context_mask = batch["context_mask"]
        target = batch["target_window"]
        target_mask = batch["target_mask"]

        s_x, _, _ = encoder_model.apply(jepa_params, context, context_mask)
        s_y, _, _ = encoder_model.apply(target_encoder_params, target, target_mask)

        # --- Raw Invariance (no predictor needed for eval) ---
        # Actually we DO need the predictor to evaluate prediction quality
        # But for pure representation quality, we measure the JEPA losses directly

        # Raw variance (no PID scaling)
        std_x = jnp.sqrt(jnp.var(s_x, axis=0) + 1e-4)
        raw_var = jnp.mean(std_x ** 2)
        var_hinge = jnp.mean(jax.nn.relu(1.0 - std_x))

        # Raw covariance
        batch_size, dim = s_x.shape
        x_mu = s_x - jnp.mean(s_x, axis=0)
        cov_x = (x_mu.T @ x_mu) / (batch_size - 1)
        off_diag_mask = 1.0 - jnp.eye(dim)
        raw_cov = jnp.sum((cov_x * off_diag_mask) ** 2) / dim

        # Topological loss
        topo = macro_topological_loss(context, s_x)

        # Stable rank
        s_vals = jnp.linalg.svd(x_mu, compute_uv=False)
        stable_rank = jnp.sum(s_vals ** 2) / (jnp.max(s_vals) ** 2 + 1e-12)

        # GP ELBO
        s_x_frozen = s_x.astype(jnp.float64)
        s_y_frozen = s_y.astype(jnp.float64)

        batch_size_gp = s_x_frozen.shape[0]
        t_context = jnp.zeros((batch_size_gp, 1), dtype=jnp.float64)
        gp_x = jnp.concatenate([s_x_frozen, t_context], axis=-1)

        gp_dataset = gpx.Dataset(X=gp_x, y=s_y_frozen)
        elbo = gpx.objectives.elbo(svgp_q, gp_dataset)

        metrics = {
            "eval_var": raw_var,
            "eval_var_dims": std_x ** 2,  # Track individual dimensions
            "eval_var_hinge": var_hinge,
            "eval_cov": raw_cov,
            "eval_topo": topo,
            "eval_stable_rank": stable_rank,
            "eval_gp_nelbo": -elbo,
        }
        return metrics

    @eqx.filter_jit
    def eval_step(
        jepa_state: Dict[str, Any], svgp_q, batch: Dict[str, jnp.ndarray]
    ):
        """Pure evaluation step — no parameter updates, no adaptive weights."""
        return eval_loss_fn(
            jepa_state["encoder_params"],
            jepa_state["target_encoder_params"],
            svgp_q,
            batch,
        )

    return eval_step
