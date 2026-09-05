import jax.numpy as jnp
from flax import linen as nn

class RevIN(nn.Module):
    """
    Reversible Instance Normalization (RevIN) layer for JAX/Flax.
    Normalizes the time-dimension independently for each feature to prevent exploding gradients
    and distribution shifts, while preserving the ability to denormalize back to real values.
    """
    num_features: int
    eps: float = 1e-5

    @nn.compact
    def __call__(self, x, mode: str, mean=None, stdev=None):
        """
        x: Array of shape (Batch, Time, Features)
        """
        # Learnable affine parameters
        gamma = self.param('gamma', nn.initializers.ones, (self.num_features,))
        beta = self.param('beta', nn.initializers.zeros, (self.num_features,))

        if mode == 'norm':
            # Calculate mean and variance over the Time dimension (axis=1)
            # keepdims=True ensures shape (Batch, 1, Features) for broadcasting
            mean = jnp.mean(x, axis=1, keepdims=True)
            var = jnp.var(x, axis=1, keepdims=True)
            stdev = jnp.sqrt(var + self.eps)
            
            x_norm = (x - mean) / stdev
            x_affine = (x_norm * gamma) + beta
            return x_affine, mean, stdev

        elif mode == 'denorm':
            if mean is None or stdev is None:
                raise ValueError("mean and stdev must be provided for denorm mode.")
            
            # Reverse the affine transformation
            x_denorm = (x - beta) / gamma
            # Reverse the normalization
            return (x_denorm * stdev) + mean
            
        else:
            raise ValueError(f"Unknown mode: {mode}")
