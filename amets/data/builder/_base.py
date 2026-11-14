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

import abc
from typing import Any, Callable

from torch.utils.data import Dataset

from amets.transform import Transformation


class DatasetBuilder(abc.ABC):
    """
    Base class for DatasetBuilders.
    """

    @abc.abstractmethod
    def build_dataset(self, *args, **kwargs):
        """
        Builds the dataset into the required file format.
        """
        ...

    @abc.abstractmethod
    def load_dataset(
        self, transform_map: dict[Any, Callable[..., Transformation]]
    ) -> Dataset:
        """
        Load the dataset.

        :param transform_map: a map which returns the required dataset transformations to be applied
        :return: the dataset ready for training
        """
        ...

