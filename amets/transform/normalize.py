#  Copyright (c) 2024, Salesforce, Inc.
#  SPDX-License-Identifier: Apache-2
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
Normalization transforms.

Implements RevIN-style normalization with context statistics.
"""

from dataclasses import dataclass
from typing import Any


from ._base import Transformation


@dataclass
class NormalizeByContext(Transformation):
    """
    RevIN-style normalization with context statistics.

    Computes the center and std of each variate from its context portion, where
    prediction_mask is False, and normalizes the whole sequence, context and
    forecast, with them. This keeps the wide value ranges of heterogeneous
    datasets numerically stable during training.

    Runs after MaskedPrediction and Patchify, so target has shape
    (var, time_patches, max_patch_size).

    Args:
        target_field: Name of the target field.
        prediction_mask_field: Name of the prediction mask field.
        eps: Small constant for numerical stability.
        subtract_last: If True, subtract the mean of the last context patch. If
            False, subtract the mean of all context values.
    """
    target_field: str = "target"
    prediction_mask_field: str = "prediction_mask"
    eps: float = 1e-5
    subtract_last: bool = False
    
    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        """
        Normalize the target with context statistics.

        Args:
            data_entry: Dictionary containing target and prediction_mask.

        Returns:
            data_entry with the normalized target.
        """
        target = data_entry[self.target_field]  # (var, time_patches, max_patch_size)
        prediction_mask = data_entry[self.prediction_mask_field]  # (var, time_patches)
        
        # The actual patch_size, used to exclude zero padding.
        patch_size = data_entry.get('patch_size', None)
        
        # Normalize each variate independently.
        num_variates = target.shape[0]
        
        for v in range(num_variates):
            # Context mask for this variate.
            if prediction_mask.ndim == 2:
                context_mask = ~prediction_mask[v]  # (time_patches,)
            else:
                context_mask = ~prediction_mask  # (time_patches,), shared by all variates
            
            context_patches = target[v, context_mask, :]  # (context_patches, max_patch_size)
            
            if len(context_patches) == 0:
                # Without context, leave this variate unnormalized.
                continue
            
            # Keep only the first patch_size values, dropping the zero padding from Patchify.
            if patch_size is not None and context_patches.shape[-1] > patch_size:
                context_patches = context_patches[:, :patch_size]  # (context_patches, patch_size)
            
            # All context values, without padding zeros.
            context_flat = context_patches.flatten()
            
            if self.subtract_last:
                # Mean of the last context patch.
                center = context_patches[-1].mean()
            else:
                # Mean of all context values.
                center = context_flat.mean()
            
            # Std of all context values.
            std = context_flat.std() + self.eps
            # Leave near-constant variates unscaled.
            if std < 0.001:
                std = 1.0
            
            # Normalize every position of this variate with the scalar statistics.
            target[v] = (target[v] - center) / std
        
        data_entry[self.target_field] = target
        return data_entry
