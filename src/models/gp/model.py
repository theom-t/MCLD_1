"""
MCLD-1 Stage 3: Sparse Variational Gaussian Process (SVGP)

Constructs a GPJax SVGP model with a configurable composite kernel.
The GP operates in a (D+1)-dimensional input space: the D latent dimensions
from the JEPA encoder plus a normalised temporal index, allowing the
Periodic kernel to discover macro-cycle frequencies.
"""
import jax
import gpjax as gpx
import jax.numpy as jnp
from src.utils.config import JepaConfig


def build_svgp(config: JepaConfig, train_n: int = 1000):
    """
    Constructs the GPJax SVGP model based on JepaConfig.

    The GP input dimensionality is (latent_dim + 1) because we concatenate
    a normalised temporal index to the latent state vector. This allows
    the Periodic kernel to learn macro-cycle frequencies.

    Args:
        config: JepaConfig containing GP architecture parameters.
        train_n: Number of training data points (for Likelihood batch scaling).

    Returns:
        A GPJax VariationalGaussian object ready for training.
    """
    # Input dim = latent_dim + 1 (temporal index)
    gp_input_dim = config.latent_dim + 1
    active_dims = list(range(gp_input_dim))

    # Separate active dims for structural vs temporal kernels
    latent_dims = list(range(config.latent_dim))
    temporal_dim = [config.latent_dim]  # The last dimension is time

    # 1. Select the base structural kernel (operates on latent dims)
    if config.gp_kernel_type == "RBF":
        base_kernel = gpx.kernels.RBF(active_dims=latent_dims)
    elif config.gp_kernel_type == "Matern12":
        base_kernel = gpx.kernels.Matern12(active_dims=latent_dims)
    elif config.gp_kernel_type == "Matern32":
        base_kernel = gpx.kernels.Matern32(active_dims=latent_dims)
    else:  # Default Matern52
        base_kernel = gpx.kernels.Matern52(active_dims=latent_dims)

    # 2. Add Periodic kernel for cyclicality (operates on temporal dim)
    if config.gp_periodic_active:
        periodic_kernel = gpx.kernels.Periodic(active_dims=temporal_dim)

        # 3. Kernel Composition
        if config.gp_kernel_composition == "multiplicative":
            kernel = base_kernel * periodic_kernel
        else:
            kernel = base_kernel + periodic_kernel
    else:
        kernel = base_kernel

    # 4. Mean Function
    if config.gp_mean_type == "Constant":
        mean_fn = gpx.mean_functions.Constant()
    else:
        mean_fn = gpx.mean_functions.Zero()

    # 5. Prior
    prior = gpx.gps.Prior(mean_function=mean_fn, kernel=kernel)

    # 6. Likelihood
    likelihood = gpx.likelihoods.Gaussian(num_datapoints=train_n)

    # 7. Posterior
    posterior = prior * likelihood

    # 8. Inducing Points in the (D+1)-dimensional input space
    # Shape: (M, D+1)
    key = jax.random.PRNGKey(42)
    z_latent = jax.random.normal(key, (config.gp_num_inducing, config.latent_dim))
    # Spread temporal index across [0, 1] for the inducing points
    key2 = jax.random.PRNGKey(43)
    z_time = jax.random.uniform(key2, (config.gp_num_inducing, 1))
    z = jnp.concatenate([z_latent, z_time], axis=-1)  # (M, D+1)

    # 9. Variational Gaussian Family (The SVGP)
    q = gpx.variational_families.VariationalGaussian(
        posterior=posterior,
        inducing_inputs=z,
    )

    return q
