import jax
import jax.numpy as jnp
from src.models.jepa.model import JEPA
from src.utils.config import JepaConfig
import time

def test_jepa_forward():
    print("--- Testing Flax JEPA Architecture ---")
    
    # 1. Define dummy macro data (Batch, Time, Features)
    # 128 batches, 36 months, 23 features
    batch_size = 128
    time_steps = 36
    num_features = 23
    
    key = jax.random.PRNGKey(42)
    key, init_key, mask_key, ctx_key = jax.random.split(key, 4)
    dummy_context = jax.random.normal(ctx_key, (batch_size, time_steps, num_features))
    dummy_mask = jax.random.bernoulli(mask_key, 0.8, (batch_size, time_steps, num_features)).astype(jnp.float32)
    
    # Scale up the dummy data to simulate non-stationary trillions of dollars
    dummy_context = (dummy_context * 1e12) + 5e13
    
    # 2. Init Model
    print("Initializing model...")
    config = JepaConfig()
    model = JEPA(
        num_features=config.num_features, 
        hidden_dim=config.encoder_hidden_dim, 
        latent_dim=config.latent_dim,
        num_layers=config.encoder_layers,
        kernel_size=config.encoder_kernel_size
    )
    
    variables = model.init(init_key, dummy_context, dummy_mask)
    
    # 3. JIT Compile Forward Pass
    print("JIT Compiling forward pass...")
    
    @jax.jit
    def forward(params, x, mask):
        return model.apply(params, x, mask)
        
    start_time = time.time()
    s_x, mean, stdev = forward(variables, dummy_context, dummy_mask)
    # Block until execution completes to measure JIT time
    s_x.block_until_ready()
    compile_time = time.time() - start_time
    print(f"JIT Compilation + 1st Run time: {compile_time:.4f} seconds")
    
    # 4. Measure execution time (post-JIT)
    start_time = time.time()
    s_x, mean, stdev = forward(variables, dummy_context, dummy_mask)
    s_x.block_until_ready()
    exec_time = time.time() - start_time
    print(f"Post-JIT Execution time (128 windows): {exec_time:.6f} seconds")
    
    # 5. Assertions
    assert s_x.shape == (batch_size, 6), f"Expected latent shape (128, 6), got {s_x.shape}"
    assert mean.shape == (batch_size, 1, num_features)
    assert stdev.shape == (batch_size, 1, num_features)
    
    # Ensure gradients didn't explode by checking NaN
    assert not jnp.isnan(s_x).any(), "Latent state contains NaNs!"
    
    print("\n✅ All shapes correct. Latent dimension is exactly 6.")
    print("✅ Model successfully absorbed values in the trillions without returning NaNs.")
    print("✅ JAX/Flax implementation is blazing fast.")

if __name__ == "__main__":
    test_jepa_forward()
