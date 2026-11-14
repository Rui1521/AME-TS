"""
Context target transform.

Saves the context portion of the target, before the prediction mask, for regime
prediction. The regime predictor computes regime_probs on GPU in forward().
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._base import Transformation


@dataclass
class SaveContextTarget(Transformation):
    """
    Save the context portion of the target for regime prediction.
    
    Extracts the context window, where prediction_mask is False, and saves it
    as a separate field. The regime predictor uses it to compute regime_probs
    on GPU in forward().

    Args:
        target_field: Name of the target field.
        prediction_mask_field: Name of the prediction mask field.
        output_field: Name of the output field for the context target.
    """
    target_field: str = "target"
    prediction_mask_field: str = "prediction_mask"
    output_field: str = "context_target"
    
    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        """
        Extract the context portion of the target for every variate.

        Keys the output by variate_id, so randomized variate indices map to the
        right context.

        Args:
            data_entry: Dictionary containing target and prediction_mask.

        Returns:
            data_entry with a context_target field that maps variate_id to arrays.
        """
        target = data_entry[self.target_field]  # (var, time, feat) or (time, feat)
        prediction_mask = data_entry[self.prediction_mask_field]  # (var, time) or (time,)
        
        # The actual patch_size, used to exclude the zero padding added by Patchify.
        # Patchify pads each patch to max_patch_size with zeros, which would pull
        # averages toward zero.
        patch_size = data_entry.get('patch_size', None)
        
        # variate_id from AddVariateIndex with randomize=True gives the randomized
        # ID of each variate.
        variate_id_field = data_entry.get('variate_id', None)
        
        # With collection_type=dict, variate_id is a dict keyed by field.
        variate_id_array = variate_id_field.get('target', None)
        if target.ndim == 3:
            # target has shape (var, time, max_patch).
            num_variates = target.shape[0]
            
            
            # Key the output by variate_id.
            context_target_dict = {}
            
            for v in range(num_variates):
                # Randomized variate ID for this variate.
                if variate_id_array is not None and variate_id_array.ndim == 2:
                    var_id = int(variate_id_array[v, 0])  # All time steps share one ID.
                else:
                    raise ValueError(f"variate_id {v} not found")
                
                # Context mask for this variate.
                if prediction_mask.ndim == 2:
                    context_mask = ~prediction_mask[v]  # (time,)
                else:
                    raise ValueError(f"prediction_mask dim {prediction_mask.ndim} is expected to be 2")
                
                # Context portion for this variate.
                context_target = target[v, context_mask, :]  # (context_time, max_patch)
                
                # Drop zero padding by slicing to the actual patch_size.
                if patch_size is not None and context_target.shape[-1] >= patch_size:
                    context_target = context_target[..., :patch_size]  # (context_time, patch_size)
                
                # Flatten to 1D for the regime predictor.
                context_target = context_target.flatten()  # (context_time * patch_size,)
                context_target_dict[var_id] = context_target.astype(np.float32)
            
            data_entry[self.output_field] = context_target_dict
        else:
            raise ValueError(f"target dim {target.ndim} is expected to be 3")
        return data_entry
