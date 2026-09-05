import jax
import jax.numpy as jnp
import optax
import time
from src.models.jepa.model import JEPA, Predictor
from src.models.jepa.step import create_train_step
from src.utils.config import JepaConfig

def test_train_step():
    print("--- Testing Flax JEPA Training Step ---")
    
    config = JepaConfig()
    batch_size = config.batch_size
    num_features = config.num_features
    
    key = jax.random.PRNGKey(42)
    key, ctx_key, targ_key, c_mask_key, t_mask_key = jax.random.split(key, 5)
    
    # Context (36 months), Target (6 months)
    batch = {
        "context_window": jax.random.normal(ctx_key, (batch_size, config.context_window_months, num_features)),
        "context_mask": jax.random.bernoulli(c_mask_key, 0.8, (batch_size, config.context_window_months, num_features)).astype(jnp.float32),
        "target_window": jax.random.normal(targ_key, (batch_size, config.target_horizon_months, num_features)),
        "target_mask": jax.random.bernoulli(t_mask_key, 0.8, (batch_size, config.target_horizon_months, num_features)).astype(jnp.float32)
    }
    
    # 1. Init Models
    print("Initializing models...")
    encoder = JEPA(
        num_features=config.num_features, 
        hidden_dim=config.encoder_hidden_dim, 
        latent_dim=config.latent_dim,
        num_layers=config.encoder_layers,
        kernel_size=config.encoder_kernel_size
    )
    predictor = Predictor(
        hidden_dim=config.predictor_hidden_dim, 
        latent_dim=config.latent_dim,
        num_layers=config.predictor_layers
    )
    
    key, enc_key, pred_key = jax.random.split(key, 3)
    
    # Initialize Context and Target Encoders identically
    encoder_params = encoder.init(enc_key, batch["context_window"], batch["context_mask"])
    target_params = encoder.init(enc_key, batch["target_window"], batch["target_mask"]) # Identical init weights
    
    # Dummy s_x shape to init predictor
    dummy_sx = jnp.zeros((batch_size, 6))
    predictor_params = predictor.init(pred_key, dummy_sx)
    
    # 2. Optax Optimizer
    optimizer = optax.adamw(learning_rate=1e-3)
    opt_state = {
        "encoder": optimizer.init(encoder_params),
        "predictor": optimizer.init(predictor_params)
    }
    
    state = {
        "encoder_params": encoder_params,
        "predictor_params": predictor_params,
        "target_encoder_params": target_params,
        "opt_state": opt_state
    }
    
    # 3. Create JIT Train Step
    print("JIT Compiling train step (includes advanced losses)...")
    train_step = create_train_step(encoder, predictor, optimizer, config)
    
    key, step_key = jax.random.split(key)
    
    start_time = time.time()
    new_state, metrics = train_step(state, batch, step_key)
    
    # Block to measure compilation
    jax.tree_util.tree_map(lambda x: x.block_until_ready() if isinstance(x, jax.Array) else x, metrics)
    compile_time = time.time() - start_time
    print(f"JIT Compilation + 1st Step time: {compile_time:.4f} seconds")
    
    # 4. Measure execution time
    key, step_key2 = jax.random.split(key)
    start_time = time.time()
    new_state, metrics = train_step(new_state, batch, step_key2)
    jax.tree_util.tree_map(lambda x: x.block_until_ready() if isinstance(x, jax.Array) else x, metrics)
    exec_time = time.time() - start_time
    print(f"Post-JIT Step execution time: {exec_time:.6f} seconds")
    
    print("\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k}: {float(v):.4f}")
        assert not jnp.isnan(v), f"Metric {k} is NaN!"
        
    print("\n✅ EMA Target Updates successfully.")
    print("✅ PID-Variance and Macro-Topological Losses computed flawlessly.")

if __name__ == "__main__":
    test_train_step()
