"""
Trend strength of a time series from the slope of a linear regression on the
min-max normalized series.
"""

import numpy as np
from scipy import stats


class TrendCalculator:
    """Computes trend strength for batched time series.
    
    Accepts only 3D input arrays of shape (batch_size, time_steps, features).
    """

    def __init__(self, ts, normalize=True):
        """
        Args:
            ts: 3D array (batch_size, time_steps, features)
            normalize: Whether to normalize the time series before computing trend.
        """
        if not isinstance(ts, np.ndarray):
            raise TypeError("Input ts must be a numpy array.")
        
        if ts.ndim != 3:
            raise ValueError(
                f"Input ts must be a 3D array with shape (batch_size, time_steps, features). "
                f"Got shape {ts.shape} with {ts.ndim} dimensions."
            )
        
        self.ts = ts
        self.normalize = normalize

    def _normalize_series(self, ts_1d):
        """
        Normalize time series to [0, 1] range.
        
        Args:
            ts_1d: 1D array (time_steps,)
            
        Returns:
            normalized_ts: 1D array (time_steps,)
        """
        ts_min = np.min(ts_1d)
        ts_max = np.max(ts_1d)
        
        if ts_max - ts_min < 1e-8:
            return np.zeros_like(ts_1d)  # Constant series
        
        return (ts_1d - ts_min) / (ts_max - ts_min)

    def _compute_trend_strength(self, ts_1d):
        """
        Computes trend strength for a single time series.
        
        With normalization, trend strength is the absolute total change of the
        fitted line, slope * length, clipped to [0, 1]. A normalized series lies
        in [0, 1], so a line that rises from 0 to 1 or falls from 1 to 0 across
        the series gives 1.0.
        
        Args:
            ts_1d: 1D array (time_steps,)
            
        Returns:
            trend_strength: float in [0, 1], where 1 means strong trend
        """
        # Handle edge cases
        if len(ts_1d) < 2:
            return 0.0  # Need at least 2 points for trend
        
        if np.std(ts_1d) < 1e-8:
            return 0.0  # Constant series has no trend
        
        try:
            # Normalize if requested
            if self.normalize:
                ts_normalized = self._normalize_series(ts_1d)
            else:
                ts_normalized = ts_1d
            
            # Fit linear regression
            x = np.arange(len(ts_normalized))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, ts_normalized)
            
            if self.normalize:
                # The slope is the change per time step, so slope * len is the
                # total change of the fitted line. It is at most about 1.0 for a
                # normalized series.
                trend_strength = abs(slope * len(ts_normalized))
                trend_strength = min(1.0, trend_strength)  # Clip to [0, 1]
            else:
                # Without normalization, use |r|, the absolute correlation with
                # time, as a direction-agnostic measure of trend strength.
                trend_strength = abs(r_value)
            
            return trend_strength
            
        except Exception as e:
            # If regression fails, return 0
            return 0.0

    def compute_trend_strength(self, mode='per_feature'):
        """
        Computes trend strength for the entire series.
        
        Args:
            mode: 
                - 'per_feature': Compute trend for each feature separately.
                - 'combined': Average features and compute trend on combined signal.
                
        Returns:
            trend_strength:
                - If mode='per_feature': 2D array (batch_size, features)
                - If mode='combined': 1D array (batch_size,)
        """
        batch_size, time_steps, features = self.ts.shape
        
        if mode == 'per_feature':
            result = np.zeros((batch_size, features))
            for b in range(batch_size):
                for f in range(features):
                    result[b, f] = self._compute_trend_strength(self.ts[b, :, f])
            return result
            
        elif mode == 'combined':
            result = np.zeros(batch_size)
            for b in range(batch_size):
                combined_signal = np.mean(self.ts[b, :, :], axis=1)
                result[b] = self._compute_trend_strength(combined_signal)
            return result
            
        else:
            raise ValueError("Mode must be 'per_feature' or 'combined'.")
