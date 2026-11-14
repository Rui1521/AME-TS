import numpy as np
from scipy.signal import windows
from scipy.stats import entropy

class Trend:
    """Trend removal using linear regression."""

    def __init__(self, ts):
        """
        Args:
            ts: 1D array (time_steps,) representing a single time series.
        """
        self.ts = ts
        self.train_length = len(ts)
        self.a, self.b = self.find_trend(ts)

    def find_trend(self, insample_data):
        """Fits linear regression y = a*x + b to the time series."""
        x = np.arange(len(insample_data))
        a, b = np.polyfit(x, insample_data, 1)
        return a, b

    def detrend(self):
        """Removes the linear trend from the time series."""
        return self.ts - (self.a * np.arange(len(self.ts)) + self.b)

class SpectralPredictabilityCalculator:
    """Computes spectral predictability for batched time series.
    
    Accepts only 3D input arrays of shape (batch_size, time_steps, features).
    """

    def __init__(self, ts, detrend=True):
        """
        Args:
            ts: 3D array (batch_size, time_steps, features)
            detrend: Whether to detrend the series before computing Spectral Entropy.
        """
        if not isinstance(ts, np.ndarray):
            raise TypeError("Input ts must be a numpy array.")
        
        if ts.ndim != 3:
            raise ValueError(
                f"Input ts must be a 3D array with shape (batch_size, time_steps, features). "
                f"Got shape {ts.shape} with {ts.ndim} dimensions."
            )
        
        self.ts = ts
        self.detrend = detrend

    def _detrend_series(self, ts_batch):
        """
        Detrends each time series in the batch.

        Args:
            ts_batch: 3D array (batch_size, time_steps, features)

        Returns:
            detrended_ts: 3D array (batch_size, time_steps, features)
        """
        batch_size, time_steps, features = ts_batch.shape
        detrended = np.zeros_like(ts_batch)

        for b in range(batch_size):
            for f in range(features):
                detrended[b, :, f] = Trend(ts_batch[b, :, f]).detrend()

        return detrended

    def _compute_spectral_predictability(self, ts):
        """
        Computes spectral predictability for a single time series.

        Args:
            ts: 1D array (time_steps,)

        Returns:
            predictability: float, 1 minus the normalized spectral entropy
        """
        window_func = windows.hann(len(ts))
        ts = ts * window_func

        ts = (ts - np.min(ts)) / (np.max(ts) - np.min(ts) + 1e-8)  # Normalize
        fourier_ts = np.abs(np.fft.rfft(ts))

        fourier_ts = (fourier_ts - np.min(fourier_ts)) / (np.max(fourier_ts) - np.min(fourier_ts) + 1e-8)
        fourier_ts /= fourier_ts.sum()
        entropy_ts = entropy(fourier_ts)
        
        n_freq_bins = len(fourier_ts)
        fore_ts = 1 - entropy_ts / np.log(n_freq_bins)
        return 0 if np.isnan(fore_ts) else fore_ts

    def compute_spectral_predictability(self, mode='per_feature'):
        """
        Computes spectral predictability for the entire series.

        Args:
            mode: 
                - 'per_feature': Compute spectral predictability for each feature separately.
                - 'combined': Average features and compute spectral predictability on the combined signal.

        Returns:
            spectral predictability:
                - If mode='per_feature': 2D array (batch_size, features)
                - If mode='combined': 1D array (batch_size,)
        """
        ts = self._detrend_series(self.ts) if self.detrend else self.ts
        batch_size, time_steps, features = ts.shape

        if mode == 'per_feature':
            result = np.zeros((batch_size, features))
            for b in range(batch_size):
                for f in range(features):
                    result[b, f] = self._compute_spectral_predictability(ts[b, :, f])
            return result

        elif mode == 'combined':
            result = np.zeros(batch_size)
            for b in range(batch_size):
                combined_signal = np.mean(ts[b, :, :], axis=1)
                result[b] = self._compute_spectral_predictability(combined_signal)
            return result

        else:
            raise ValueError("Mode must be 'per_feature' or 'combined'.")
