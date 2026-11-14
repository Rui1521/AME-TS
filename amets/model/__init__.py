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
AMETS: a time-series foundation model with structure-aware routing.

The package builds a prior over experts from predicted structural descriptors
and uses it to guide expert selection in the MoE layers. The model predicts
points and trains with an L1 or L2 loss.
"""

from .combined_regime_predictor import CombinedRegimePredictor
from .module import AMETSModule
from .pretrain import AMETSPretrain

__all__ = [
    "AMETSModule",
    "AMETSPretrain",
    "CombinedRegimePredictor",
]
