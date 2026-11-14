"""
Feature extraction for the four structural descriptors.

TimeSeriesFeatureExtractor computes all four descriptors through one interface:
1. Forecastability, from spectral predictability.
2. Seasonality strength, from an STL decomposition.
3. Trend strength, from a normalized linear-regression slope.
4. Sparsity, as the ratio of near-zero values.
"""

import numpy as np
from .spectral_predictability import SpectralPredictabilityCalculator
from .seasonality import SeasonalityCalculator
from .trend import TrendCalculator
from .sparsity import SparsityCalculator


class TimeSeriesFeatureExtractor:
    """
    Computes the four structural descriptors of a batch of time series.
    
    Accepts only 3D input arrays of shape (batch_size, time_steps, features).
    
    The descriptors are:
    - Forecastability: how predictable the series is.
    - Seasonality: the strength of periodic patterns.
    - Trend: the strength of directional movement.
    - Sparsity: the ratio of near-zero values.
    """
    
    def __init__(self, ts, period=None, zero_threshold=1e-8):
        """
        Args:
            ts: 3D array (batch_size, time_steps, features)
            period: Seasonal period for the seasonality descriptor. None auto-detects it.
            zero_threshold: Values with absolute value below this count as zero for sparsity.
        """
        if not isinstance(ts, np.ndarray):
            raise TypeError("Input ts must be a numpy array.")
        
        if ts.ndim != 3:
            raise ValueError(
                f"Input ts must be a 3D array with shape (batch_size, time_steps, features). "
                f"Got shape {ts.shape} with {ts.ndim} dimensions."
            )
        
        self.ts = ts
        self.period = period
        self.zero_threshold = zero_threshold
        
        # One calculator per descriptor
        self.forecastability_calc = SpectralPredictabilityCalculator(ts, detrend=True)
        self.seasonality_calc = SeasonalityCalculator(ts, period=period)
        self.trend_calc = TrendCalculator(ts, normalize=True)
        self.sparsity_calc = SparsityCalculator(ts, zero_threshold=zero_threshold)
    
    def compute_all_features(self, mode='combined'):
        """
        Compute all four descriptors for the time series.
        
        Args:
            mode: 'per_feature' or 'combined'
                - 'per_feature': Compute descriptors for each channel separately.
                - 'combined': Average the channels and compute descriptors on the result.
        
        Returns:
            features: Dictionary containing all four descriptors
                - 'forecastability': Spectral predictability scores
                - 'seasonality': Seasonality strength scores
                - 'trend': Trend strength scores
                - 'sparsity': Sparsity scores
                
            Shape depends on mode:
                - If mode='per_feature': Each value is 2D array (batch_size, features)
                - If mode='combined': Each value is 1D array (batch_size,)
        """
        features = {
            'forecastability': self.forecastability_calc.compute_spectral_predictability(mode=mode),
            'seasonality': self.seasonality_calc.compute_seasonality_strength(mode=mode),
            'trend': self.trend_calc.compute_trend_strength(mode=mode),
            'sparsity': self.sparsity_calc.compute_sparsity(mode=mode)
        }
        
        return features
