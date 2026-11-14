"""
Sparsity as the ratio of zero or near-zero values in a time series.
"""

import numpy as np


class SparsityCalculator:
    """Computes sparsity for batched time series.
    
    Accepts only 3D input arrays of shape (batch_size, time_steps, features).
    """

    def __init__(self, ts, zero_threshold=1e-8):
        """
        Args:
            ts: 3D array (batch_size, time_steps, features)
            zero_threshold: Values with absolute value below this threshold are considered zero.
        """
        if not isinstance(ts, np.ndarray):
            raise TypeError("Input ts must be a numpy array.")
        
        if ts.ndim != 3:
            raise ValueError(
                f"Input ts must be a 3D array with shape (batch_size, time_steps, features). "
                f"Got shape {ts.shape} with {ts.ndim} dimensions."
            )
        
        self.ts = ts
        self.zero_threshold = zero_threshold

    def _compute_sparsity(self, ts_1d):
        """
        Computes sparsity for a single time series.
        
        Sparsity is defined as the ratio of zero (or near-zero) values
        to the total number of values.
        
        Args:
            ts_1d: 1D array (time_steps,)
            
        Returns:
            sparsity: float in [0, 1], where 1 means completely sparse (all zeros)
        """
        if len(ts_1d) == 0:
            return 0.0
        
        # Count values that are effectively zero
        zero_count = np.sum(np.abs(ts_1d) < self.zero_threshold)
        
        # Compute ratio
        sparsity = zero_count / len(ts_1d)
        
        return sparsity

    def compute_sparsity(self, mode='per_feature'):
        """
        Computes sparsity for the entire series.
        
        Args:
            mode: 
                - 'per_feature': Compute sparsity for each feature separately.
                - 'combined': Compute sparsity across all features combined.
                
        Returns:
            sparsity:
                - If mode='per_feature': 2D array (batch_size, features)
                - If mode='combined': 1D array (batch_size,)
        """
        batch_size, time_steps, features = self.ts.shape
        
        if mode == 'per_feature':
            result = np.zeros((batch_size, features))
            for b in range(batch_size):
                for f in range(features):
                    result[b, f] = self._compute_sparsity(self.ts[b, :, f])
            return result
            
        elif mode == 'combined':
            result = np.zeros(batch_size)
            for b in range(batch_size):
                # Flatten all features and compute overall sparsity
                flattened = self.ts[b, :, :].flatten()
                result[b] = self._compute_sparsity(flattened)
            return result
            
        else:
            raise ValueError("Mode must be 'per_feature' or 'combined'.")
