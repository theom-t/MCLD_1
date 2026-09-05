import jax.numpy as jnp
from flax import linen as nn

from .revin import RevIN
from .encoder import ContextEncoder

class Predictor(nn.Module):
    """
    Residual MLP Predictor.
    Predicts the future latent state from the current latent state.
    """
    hidden_dim: int = 32
    latent_dim: int = 6
    num_layers: int = 2

    @nn.compact
    def __call__(self, x):
        h = x
        for _ in range(self.num_layers):
            h = nn.Dense(features=self.hidden_dim)(h)
            h = nn.LayerNorm()(h)
            h = nn.gelu(h)
            
        out = nn.Dense(features=self.latent_dim)(h)
        # Residual connection
        return x + out

class JEPA(nn.Module):
    """
    The Temporal Joint-Embedding Predictive Architecture.
    Orchestrates the normalization, encoding, and prediction phases.
    """
    num_features: int = 23
    hidden_dim: int = 64
    latent_dim: int = 6
    num_layers: int = 3
    kernel_size: int = 3

    @nn.compact
    def __call__(self, x, mask):
        """
        Standard forward pass (used for inference/deployment).
        x: Raw macro features (Batch, Time, 23)
        mask: Binary indicator of observed data (Batch, Time, 23)
        Returns the latent state and the normalization statistics.
        """
        # 1. Normalize Context (Only normalize the macro features, NOT the mask)
        x_norm, mean, stdev = RevIN(num_features=self.num_features)(x, mode='norm')
        
        # 2. Concatenate the normalized features with their binary masks
        # Shape goes from (Batch, 36, 23) -> (Batch, 36, 46)
        x_combined = jnp.concatenate([x_norm, mask], axis=-1)
        
        # 3. Extract Latent State
        s_x = ContextEncoder(
            hidden_dim=self.hidden_dim, 
            latent_dim=self.latent_dim,
            num_layers=self.num_layers,
            kernel_size=self.kernel_size
        )(x_combined)
        
        return s_x, mean, stdev
