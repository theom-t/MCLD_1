import jax
import jax.numpy as jnp
import optax
import os
import equinox as eqx

from src.utils.config import JepaConfig
from src.data.window_generator import MacroDataBridge
from src.models.jepa.model import JEPA, Predictor
from src.models.gp.model import build_svgp
from src.pipeline.train_step import create_joint_train_step

def train_foundation_model():
    print(f"\n{'='*60}")
    print(f"🚀 TRAINING MCLD-1 STAGE 2 FOUNDATION MODEL")
    print(f"{'='*60}")
    
    # Target exactly the peak epoch for Trial 250
    max_epochs = 124
    
    config = JepaConfig()
    
    data_bridge = MacroDataBridge(config, "data/processed")
    data_bridge.build_datasets()
    
    input_dim = data_bridge.datasets["train"]["context_window"].shape[-1]
    config.num_features = input_dim
    
    # Initialize
    key = jax.random.PRNGKey(250)
    key, enc_key, pred_key, train_key = jax.random.split(key, 4)
    
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
    
    # Optimizers
    jepa_optimizer = optax.adamw(learning_rate=config.learning_rate, weight_decay=config.weight_decay)
    gp_optimizer = optax.adam(learning_rate=config.gp_learning_rate)
    
    opt_state_jepa = {
        "encoder": jepa_optimizer.init(encoder_params),
        "predictor": jepa_optimizer.init(predictor_params),
    }
    gp_opt_state = gp_optimizer.init(eqx.filter(svgp_q, eqx.is_inexact_array))
    
    jepa_state = {
        "encoder_params": encoder_params,
        "predictor_params": predictor_params,
        "target_encoder_params": target_params,
        "opt_state": opt_state_jepa,
    }
    
    train_step = create_joint_train_step(encoder, predictor, jepa_optimizer, config)
    
    # Training Loop
    print(f"Training Trial 250 configuration for exactly {max_epochs} epochs to hit peak Stable Rank...")
    for epoch in range(max_epochs):
        train_gen = data_bridge.get_batch_generator("train", config.batch_size, shuffle=True)
        
        for batch in train_gen:
            train_key, step_key = jax.random.split(train_key)
            jepa_state, gp_opt_state, svgp_q, _ = train_step(
                jepa_state, gp_opt_state, gp_optimizer, svgp_q, batch, step_key
            )
        
        if epoch % 10 == 0 or epoch == max_epochs - 1:
            print(f"Epoch {epoch:03d} / {max_epochs} complete.")
            
    # Save Final Foundation Model Weights
    os.makedirs("checkpoints/foundation", exist_ok=True)
    eqx.tree_serialise_leaves("checkpoints/foundation/stage2_encoder.eqx", jepa_state["encoder_params"])
    eqx.tree_serialise_leaves("checkpoints/foundation/stage2_predictor.eqx", jepa_state["predictor_params"])
    
    print("\n✅ Stage 2 Foundation Model fully trained and locked in!")
    print("Weights saved to checkpoints/foundation/stage2_encoder.eqx")

if __name__ == "__main__":
    train_foundation_model()
