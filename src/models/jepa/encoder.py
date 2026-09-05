import jax.numpy as jnp
from flax import linen as nn

class CausalConvBlock(nn.Module):
    """
    A 1D Causal Convolution block.
    Asymmetric padding on the left ensures the convolution only looks at past and present data.
    """
    features: int
    kernel_size: int
    dilation: int

    @nn.compact
    def __call__(self, x):
        # Calculate padding required to maintain causality and sequence length
        pad_amount = (self.kernel_size - 1) * self.dilation
        
        # Pad ONLY the past (left side of the time axis, which is axis 1)
        # padding format: ((batch_pad), (time_pad), (feature_pad))
        x_padded = jnp.pad(x, ((0, 0), (pad_amount, 0), (0, 0)))
        
        # 1D Convolution over the padded sequence using 'VALID' so the output length matches input length
        y = nn.Conv(
            features=self.features, 
            kernel_size=(self.kernel_size,), 
            kernel_dilation=(self.dilation,),
            padding='VALID'
        )(x_padded)
        
        y = nn.LayerNorm()(y)
        return nn.gelu(y)

class ContextEncoder(nn.Module):
    """
    Causal TCN Encoder.
    Compresses a continuous window of multidimensional features into a low-dimensional latent state.
    """
    hidden_dim: int = 64
    latent_dim: int = 6
    num_layers: int = 3
    kernel_size: int = 3

    @nn.compact
    def __call__(self, x):
        # Dynamically build causal conv blocks with expanding dilation
        for i in range(self.num_layers):
            dilation = 2 ** i
            x = CausalConvBlock(
                features=self.hidden_dim, 
                kernel_size=self.kernel_size, 
                dilation=dilation
            )(x)
        
        # Take the final temporal state as the summary embedding for the entire context window
        # Shape goes from (Batch, Time, Hidden) to (Batch, Hidden)
        x_final = x[:, -1, :] 
        
        # Project to the dense latent dimension
        latent = nn.Dense(features=self.latent_dim)(x_final)
        return latent
