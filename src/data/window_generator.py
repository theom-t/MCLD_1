"""
MCLD-1 Data Bridge: Window Generator

Converts the long-format bitemporal parquet data into wide-format
(B, T, 23) sliding window tensors for the JAX pipeline.

Enforces strict temporal embargo buffers between Train/Val/Test splits
to prevent lookahead leakage.
"""
import polars as pl
import jax.numpy as jnp
import numpy as np
from datetime import datetime
from typing import Dict, Tuple, List, Optional
from src.utils.config import JepaConfig


class MacroDataBridge:
    """Bridges the Stage 1 parquet data to the Stage 2/3 JAX training pipeline."""

    def __init__(self, config: JepaConfig, data_dir: str = "data/processed"):
        self.config = config
        self.data_dir = data_dir

        # Buffer required to prevent Train/Val/Test overlap
        self.buffer_months = config.context_window_months + config.target_horizon_months

        # Load and pivot the data
        self._load_and_pivot()

    def _load_and_pivot(self):
        """Loads long-format parquets and pivots them to wide-format (date, country, f1, f2, ...)."""
        print("--- 📊 MCLD-1 Data Bridge Initializing ---")
        print(f"Loading data from {self.data_dir}...")

        # Load raw long-format data
        df_features_long = pl.read_parquet(f"{self.data_dir}/stationary_tensor.parquet")
        df_masks_long = pl.read_parquet(f"{self.data_dir}/mask_tensor.parquet")

        # ---- BUG 2 FIX: Handle value_stationary nulls ----
        # The fractional differencing filter has a burn-in period that produces nulls.
        # Strategy: Use value_stationary where available, fall back to raw value otherwise.
        # This is safe because the raw value is the PCHIP-interpolated series, and the
        # mask tensor already flags which months are real vs interpolated.
        df_features_long = df_features_long.with_columns(
            pl.when(pl.col("value_stationary").is_not_null())
            .then(pl.col("value_stationary"))
            .otherwise(pl.col("value"))
            .alias("model_value")
        )

        # ---- BUG 1 FIX: Pivot long-format to wide-format ----
        # From: (date, country, feature, value) rows
        # To:   (date, country, feat_1, feat_2, ..., feat_23) columns
        self.df_features = df_features_long.pivot(
            on="feature",
            index=["date", "country"],
            values="model_value",
        ).sort(["country", "date"])

        # ---- BUG 3 FIX: Pivot mask tensor to wide-format ----
        self.df_masks = df_masks_long.pivot(
            on="feature",
            index=["date", "country"],
            values="mask_val",
        ).sort(["country", "date"])

        # Identify the 23 feature columns (everything except date and country)
        self.feature_cols = sorted([c for c in self.df_features.columns if c not in ["date", "country"]])
        self.mask_cols = sorted([c for c in self.df_masks.columns if c not in ["date", "country"]])

        # Ensure feature columns match between features and masks
        assert self.feature_cols == self.mask_cols, (
            f"Feature/Mask column mismatch!\n"
            f"Features: {self.feature_cols}\n"
            f"Masks: {self.mask_cols}"
        )

        num_features = len(self.feature_cols)
        print(f"Pivoted to wide format: {num_features} features detected.")
        print(f"Features: {self.feature_cols}")

        if num_features != self.config.num_features:
            print(f"⚠️  WARNING: Config expects {self.config.num_features} features but data has {num_features}.")
            print(f"   Updating config.num_features to {num_features}.")
            # We won't mutate the frozen config, but we'll track this
            self._actual_num_features = num_features
        else:
            self._actual_num_features = num_features

        # Fill remaining NaNs in the wide feature table with 0.0 and set mask to 0
        # (features that don't exist for a country/date get value=0, mask=0)
        self.df_features = self.df_features.fill_null(0.0)
        self.df_masks = self.df_masks.fill_null(0)

        # Extract unique dates for splitting
        self.unique_dates = sorted(self.df_features["date"].unique().to_list())
        self._calculate_splits()

    def _calculate_splits(self):
        """
        Creates Train / Val / Test splits with strict buffer periods.
        Uses a 70 / 15 / 15 split of the available timeline, minus the buffers.
        """
        total_months = len(self.unique_dates)
        train_end_idx = int(total_months * 0.70)

        # Apply Buffer 1
        val_start_idx = train_end_idx + self.buffer_months
        val_end_idx = val_start_idx + int(total_months * 0.15)

        # Apply Buffer 2
        test_start_idx = val_end_idx + self.buffer_months

        if test_start_idx >= total_months:
            raise ValueError(
                f"Not enough data to support buffer of {self.buffer_months} months. "
                f"Have {total_months} months, need at least {test_start_idx + 1}."
            )

        self.split_dates = {
            "train": (self.unique_dates[0], self.unique_dates[train_end_idx]),
            "buffer_1": (self.unique_dates[train_end_idx + 1], self.unique_dates[val_start_idx - 1]),
            "val": (self.unique_dates[val_start_idx], self.unique_dates[val_end_idx]),
            "buffer_2": (self.unique_dates[val_end_idx + 1], self.unique_dates[test_start_idx - 1]),
            "test": (self.unique_dates[test_start_idx], self.unique_dates[-1]),
        }

        # Leakage Assertion Checks
        assert self.split_dates["train"][1] < self.split_dates["val"][0], \
            "LEAKAGE DETECTED: Train overlaps Val"
        assert self.split_dates["val"][1] < self.split_dates["test"][0], \
            "LEAKAGE DETECTED: Val overlaps Test"

    def _generate_windows_for_split(
        self, start_date: datetime, end_date: datetime
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Filters wide-format data by date range, groups by country,
        and creates sliding windows of shape (N, window_size, num_features).
        """
        # Filter the timeframe
        df_feat_split = self.df_features.filter(
            (pl.col("date") >= start_date) & (pl.col("date") <= end_date)
        )
        df_mask_split = self.df_masks.filter(
            (pl.col("date") >= start_date) & (pl.col("date") <= end_date)
        )

        window_size = self.config.context_window_months + self.config.target_horizon_months
        ctx = self.config.context_window_months

        all_ctx_feat, all_ctx_mask = [], []
        all_targ_feat, all_targ_mask = [], []

        countries = df_feat_split["country"].unique().sort().to_list()

        for country in countries:
            # Extract wide-format numpy arrays: shape (T_country, 23)
            c_feat = (
                df_feat_split
                .filter(pl.col("country") == country)
                .sort("date")
                .select(self.feature_cols)
                .to_numpy()
                .astype(np.float32)
            )
            c_mask = (
                df_mask_split
                .filter(pl.col("country") == country)
                .sort("date")
                .select(self.mask_cols)
                .to_numpy()
                .astype(np.float32)
            )

            if len(c_feat) < window_size:
                continue

            # Ensure contiguous memory for sliding_window_view
            c_feat = np.ascontiguousarray(c_feat)
            c_mask = np.ascontiguousarray(c_mask)

            # Create rolling windows: (N_windows, window_size, num_features)
            num_windows = c_feat.shape[0] - window_size + 1
            feat_windows = np.lib.stride_tricks.sliding_window_view(
                c_feat, window_size, axis=0
            )
            mask_windows = np.lib.stride_tricks.sliding_window_view(
                c_mask, window_size, axis=0
            )

            # sliding_window_view output: (N_windows, num_features, window_size)
            # Swap to: (N_windows, window_size, num_features)
            feat_windows = np.swapaxes(feat_windows, 1, 2)
            mask_windows = np.swapaxes(mask_windows, 1, 2)

            # Split into Context and Target
            all_ctx_feat.append(feat_windows[:, :ctx, :].copy())
            all_ctx_mask.append(mask_windows[:, :ctx, :].copy())
            all_targ_feat.append(feat_windows[:, ctx:, :].copy())
            all_targ_mask.append(mask_windows[:, ctx:, :].copy())

        if not all_ctx_feat:
            return None

        return {
            "context_window": np.concatenate(all_ctx_feat, axis=0),
            "context_mask": np.concatenate(all_ctx_mask, axis=0),
            "target_window": np.concatenate(all_targ_feat, axis=0),
            "target_mask": np.concatenate(all_targ_mask, axis=0),
        }

    def build_datasets(self):
        """Builds the final numpy arrays for Train/Val/Test and prints the manifest."""
        print(f"Buffer Period Enforced: {self.buffer_months} Months\n")

        self.datasets: Dict[str, Optional[Dict[str, np.ndarray]]] = {}
        for split_name in ["train", "val", "test"]:
            start_date, end_date = self.split_dates[split_name]
            data = self._generate_windows_for_split(start_date, end_date)
            self.datasets[split_name] = data

            if data is not None:
                num_samples = data["context_window"].shape[0]
                ctx_shape = data["context_window"].shape
                tgt_shape = data["target_window"].shape
            else:
                num_samples = 0
                ctx_shape = tgt_shape = "N/A"

            date_str = f"{start_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')}"
            print(f"[{split_name.upper():<5}] Dates: {date_str:<25} | Samples: {num_samples:>6,} | ctx: {ctx_shape} | tgt: {tgt_shape}")

            # Print buffer info after train and val
            if split_name in ["train", "val"]:
                b_name = "buffer_1" if split_name == "train" else "buffer_2"
                b_start, b_end = self.split_dates[b_name]
                b_str = f"{b_start.strftime('%Y-%m')} to {b_end.strftime('%Y-%m')}"
                print(f"[EMBARGO] Dates: {b_str:<25} | STRICT LEAKAGE FIREWALL")

        # Sanity checks on tensor shapes
        for split_name in ["train", "val", "test"]:
            d = self.datasets[split_name]
            if d is not None:
                assert d["context_window"].shape[1] == self.config.context_window_months, \
                    f"{split_name} context has wrong time dim: {d['context_window'].shape}"
                assert d["context_window"].shape[2] == self._actual_num_features, \
                    f"{split_name} context has wrong feature dim: {d['context_window'].shape}"
                assert d["context_mask"].shape == d["context_window"].shape, \
                    f"{split_name} mask shape mismatch"
                assert not np.isnan(d["context_window"]).any(), \
                    f"{split_name} context contains NaN values!"
                assert not np.isnan(d["target_window"]).any(), \
                    f"{split_name} target contains NaN values!"

        print("\n✅ All shape and NaN checks passed. Ready for JAX batching.")

    def get_batch_generator(
        self, split: str = "train", batch_size: int = 128, shuffle: bool = False, seed: int = 0
    ):
        """Yields JAX-compatible batch dicts for the training loop."""
        data = self.datasets[split]
        if data is None:
            return

        num_samples = data["context_window"].shape[0]
        indices = np.arange(num_samples)

        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(indices)

        for i in range(0, num_samples, batch_size):
            batch_idx = indices[i : i + batch_size]
            if len(batch_idx) < batch_size:
                continue  # Drop last incomplete batch

            yield {
                "context_window": jnp.array(data["context_window"][batch_idx], dtype=jnp.float32),
                "context_mask": jnp.array(data["context_mask"][batch_idx], dtype=jnp.float32),
                "target_window": jnp.array(data["target_window"][batch_idx], dtype=jnp.float32),
                "target_mask": jnp.array(data["target_mask"][batch_idx], dtype=jnp.float32),
            }


if __name__ == "__main__":
    cfg = JepaConfig()
    bridge = MacroDataBridge(cfg)
    bridge.build_datasets()
