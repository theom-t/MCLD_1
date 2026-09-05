"""
MCLD-1 Stage 2: Temporal-JEPA Training Step

Implements the advanced self-supervised losses for the JEPA model:
1. Adaptive VICReg (PID-controlled variance and covariance)
2. Macro-Topological Loss (preserves input space topology in latent space)
3. Data Augmentation (Gaussian Noise + Stochastic Feature Dropout)
"""
import jax
import jax.numpy as jnp
import optax
import equinox as eqx
from typing import Dict, Tuple, Any

def vicreg_loss(
    s_x: jnp.ndarray, 
    s_y: jnp.ndarray, 
    base_var_weight: float = 25.0, 
    cov_weight: float = 1.0,
    target_std: float = 1.0
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    batch_size, dim = s_x.shape
    
    # 1. Variance Loss (Hinge loss on standard deviation)
    std_x = jnp.sqrt(jnp.var(s_x, axis=0) + 1e-4)
    std_y = jnp.sqrt(jnp.var(s_y, axis=0) + 1e-4)
    
    # Standard VICReg hinge loss
    loss_var_x = jnp.mean(jax.nn.relu(target_std - std_x))
    loss_var_y = jnp.mean(jax.nn.relu(target_std - std_y))
    loss_var = loss_var_x + loss_var_y
    
    # PID-like Adaptive Variance Weighting
    # If the variance collapses, the weight aggressively spikes to force it open
    batch_variance = jnp.mean(std_x**2)
    collapse_penalty = jnp.exp(-batch_variance)  # Exponential scaling as variance approaches 0
    adaptive_var_weight = base_var_weight * (1.0 + 5.0 * collapse_penalty)
    
    loss_var = loss_var * adaptive_var_weight
    
    # 2. Covariance Loss (Decorrelate dimensions)
    x_mu = s_x - jnp.mean(s_x, axis=0)
    y_mu = s_y - jnp.mean(s_y, axis=0)
    
    cov_x = (x_mu.T @ x_mu) / (batch_size - 1)
    cov_y = (y_mu.T @ y_mu) / (batch_size - 1)
    
    # Mask out the diagonal
    off_diag_mask = 1.0 - jnp.eye(dim)
    
    loss_cov_x = jnp.sum((cov_x * off_diag_mask)**2) / dim
    loss_cov_y = jnp.sum((cov_y * off_diag_mask)**2) / dim
    loss_cov = cov_weight * (loss_cov_x + loss_cov_y)
    
    return loss_var, loss_cov, batch_variance

def macro_topological_loss(context_window: jnp.ndarray, s_x: jnp.ndarray) -> jnp.ndarray:
    batch_size = s_x.shape[0]
    
    # Flatten the context window (B, T, F) -> (B, T*F)
    flat_context = context_window.reshape(batch_size, -1)
    
    # Compute pairwise cosine similarity in the input space
    x_norm = flat_context / jnp.maximum(jnp.linalg.norm(flat_context, axis=-1, keepdims=True), 1e-12)
    input_sim = x_norm @ x_norm.T  # (B, B)
    
    # Compute pairwise cosine similarity in the latent space
    s_norm = s_x / jnp.maximum(jnp.linalg.norm(s_x, axis=-1, keepdims=True), 1e-12)
    latent_sim = s_norm @ s_norm.T  # (B, B)
    
    # MSE between the similarity matrices
    return jnp.mean((input_sim - latent_sim)**2)

def create_train_step(encoder_model, predictor_model, optimizer, config: Any):
    """
    Factory function that returns a JIT-compiled training step function.
    """

    def loss_fn(jepa_params, predictor_params, target_encoder_params, batch: Dict[str, jnp.ndarray], step_key: jax.Array):
        # Unpack batch
        context = batch["context_window"]       # (B, T_ctx, F)
        context_mask = batch["context_mask"]    # (B, T_ctx, F)
        target = batch["target_window"]         # (B, T_tgt, F)
        target_mask = batch["target_mask"]      # (B, T_tgt, F)
        
        # --- DATA AUGMENTATION (Applied to Context Only) ---
        if config.aug_gaussian_noise > 0.0 or config.aug_feature_dropout > 0.0:
            noise_key, drop_key = jax.random.split(step_key)
            
            if config.aug_gaussian_noise > 0.0:
                noise = jax.random.normal(noise_key, context.shape) * config.aug_gaussian_noise
                context = context + (noise * context_mask)
                
            if config.aug_feature_dropout > 0.0:
                keep_prob = 1.0 - config.aug_feature_dropout
                drop_mask = jax.random.bernoulli(drop_key, p=keep_prob, shape=context.shape).astype(jnp.float32)
                context = context * drop_mask
                context_mask = context_mask * drop_mask

        # Forward pass Context
        s_x, _, _ = encoder_model.apply(jepa_params, context, context_mask)
        hat_s_y = predictor_model.apply(predictor_params, s_x)

        # Forward pass Target (stop_gradient)
        s_y, _, _ = encoder_model.apply(target_encoder_params, target, target_mask)
        s_y = jax.lax.stop_gradient(s_y)

        # --- Losses ---
        # 1. Cosine Invariance
        pred_norm = hat_s_y / jnp.maximum(jnp.linalg.norm(hat_s_y, axis=-1, keepdims=True), 1e-12)
        targ_norm = s_y / jnp.maximum(jnp.linalg.norm(s_y, axis=-1, keepdims=True), 1e-12)
        loss_inv = jnp.mean(2.0 - 2.0 * jnp.sum(pred_norm * targ_norm, axis=-1))

        # 2. Adaptive VICReg Variance/Covariance
        loss_var, loss_cov, batch_var = vicreg_loss(
            s_x, s_y, 
            base_var_weight=config.loss_var_base_weight,
            cov_weight=config.loss_cov_weight
        )

        # 3. Macro-Topological Loss
        loss_topo = macro_topological_loss(context, s_x)

        # Combine
        total_loss = (
            (config.loss_inv_weight * loss_inv) + 
            loss_var + 
            loss_cov + 
            (config.loss_topo_weight * loss_topo)
        )

        metrics = {
            "loss": total_loss,
            "loss_inv": loss_inv,
            "loss_var": loss_var,
            "loss_cov": loss_cov,
            "loss_topo": loss_topo,
            "batch_variance": batch_var
        }
        return total_loss, metrics

    def wrapped_loss_fn(trainable_params, target_encoder_params, batch, step_key):
        j_params, p_params = trainable_params
        return loss_fn(j_params, p_params, target_encoder_params, batch, step_key)

    @eqx.filter_jit
    def train_step(state: Dict[str, Any], batch: Dict[str, jnp.ndarray], step_key: jax.Array):
        jepa_params = state["encoder_params"]
        predictor_params = state["predictor_params"]
        target_encoder_params = state["target_encoder_params"]
        opt_state = state["opt_state"]

        trainable = (jepa_params, predictor_params)
        
        grad_fn = eqx.filter_value_and_grad(wrapped_loss_fn, has_aux=True)
        (loss, metrics), grads = grad_fn(trainable, target_encoder_params, batch, step_key)
        
        jepa_grads, predictor_grads = grads

        # Update Encoder
        enc_updates, new_opt_state_enc = optimizer.update(
            jepa_grads, opt_state["encoder"], jepa_params
        )
        new_jepa_params = optax.apply_updates(jepa_params, enc_updates)

        # Update Predictor
        pred_updates, new_opt_state_pred = optimizer.update(
            predictor_grads, opt_state["predictor"], predictor_params
        )
        new_predictor_params = optax.apply_updates(predictor_params, pred_updates)

        # EMA update Target Encoder
        tau = config.ema_tau_base
        new_target_params = jax.tree_util.tree_map(
            lambda targ, ctx: tau * targ + (1.0 - tau) * ctx,
            target_encoder_params, new_jepa_params
        )

        new_state = {
            "encoder_params": new_jepa_params,
            "predictor_params": new_predictor_params,
            "target_encoder_params": new_target_params,
            "opt_state": {
                "encoder": new_opt_state_enc,
                "predictor": new_opt_state_pred
            }
        }
        
        return new_state, metrics

    return train_step
