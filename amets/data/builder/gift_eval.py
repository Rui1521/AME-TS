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
Pre-training dataset builder.

Loads the pre-training pools under $PRETRAIN_DATA: the GIFT-Eval train split,
synthetic data, LOTSA, and datasets from the Chronos training corpus. Every pool is already stored in
HuggingFace Arrow format, so it loads without conversion.
"""

from collections import defaultdict
from functools import partial
from pathlib import Path

from amets.data.dataset import TimeSeriesDataset

from .lotsa_v1._base import LOTSADatasetBuilder


# Dataset names accepted by the builder. Slash notation ("electricity/H") maps
# to nested directories under storage_path.
GIFT_EVAL_DATASETS = [
    # M4 Competition datasets (no frequency suffix)
    "m4_yearly",
    "m4_quarterly", 
    "m4_monthly",
    "m4_weekly",
    "m4_daily",
    "m4_hourly",
    
    # Electricity datasets (multiple frequencies)
    "electricity/15T",
    "electricity/H",
    "electricity/D",
    "electricity/W",
    
    # ETT datasets (multiple frequencies)
    "ett1/15T",
    "ett1/H",
    "ett1/D",
    "ett1/W",
    "ett2/15T",
    "ett2/H",
    "ett2/D",
    "ett2/W",
    
    # Solar datasets (multiple frequencies)
    "solar/10T",
    "solar/H",
    "solar/D",
    "solar/W",
    
    # Weather datasets
    "jena_weather/10T",
    "jena_weather/H",
    "jena_weather/D",
    
    # Traffic datasets
    "LOOP_SEATTLE/5T",
    "LOOP_SEATTLE/H",
    "LOOP_SEATTLE/D",
    "SZ_TAXI/15T",
    "SZ_TAXI/H",
    
    # Dense multivariate datasets
    "M_DENSE/H",
    "M_DENSE/D",
    
    # Cloud/IT infrastructure datasets
    "bitbrains_fast_storage/5T",
    "bitbrains_fast_storage/H",
    "bitbrains_rnd/5T",
    "bitbrains_rnd/H",
    "bizitobs_application",
    "bizitobs_l2c/5T",
    "bizitobs_l2c/H",
    "bizitobs_service",
    
    # Retail/Sales datasets
    "hierarchical_sales/D",
    "hierarchical_sales/W",
    "restaurant",
    "car_parts_with_missing",
    
    # Healthcare
    "hospital",
    
    # Environmental/Weather
    "temperature_rain_with_missing",
    "saugeenday/D",
    "saugeenday/W",
    "saugeenday/M",
    "us_births/D",
    "us_births/W",
    "us_births/M",
    
    # Other
    "covid_deaths",
    "kdd_cup_2018_with_missing/H",
    "kdd_cup_2018_with_missing/D",

    # Synthetic (diverse regime patterns)
    "synthetic",

    # M5 aggregation levels (blocked from training by pool_guard)
    "m5_L1_Total",
    "m5_L2_State",
    "m5_L3_Store",
    "m5_L4_Category",
    "m5_L5_Department",
    "m5_L6_State_Category",
    "m5_L7_State_Department",
    "m5_L8_Store_Category",
    "m5_L9_Store_Department",
    "m5_L10_Item",
    "m5_L11_State_Item",
    "m5_L12_Bottom",

    # LOTSA datasets, group 1
    "lotsa_traffic_hourly",
    "lotsa_PEMS04",
    "lotsa_weather",
    "lotsa_oikolab_weather",
    "lotsa_subseasonal",
    "lotsa_m5",
    "lotsa_favorita_sales",
    "lotsa_monash_m3_monthly",
    "lotsa_tourism_monthly",
    "lotsa_fred_md",
    "lotsa_nn5_daily_with_missing",
    "lotsa_australian_electricity_demand",
    "lotsa_wind_power",
    "lotsa_alibaba_cluster_trace_2018",
    # LOTSA datasets, group 2
    "lotsa_PEMS03",
    "lotsa_PEMS07",
    "lotsa_PEMS08",
    "lotsa_PEMS_BAY",
    "lotsa_london_smart_meters_with_missing",
    "lotsa_extended_web_traffic_with_missing",
    "lotsa_tourism_quarterly",
    "lotsa_tourism_yearly",
    "lotsa_m1_monthly",
    "lotsa_m1_quarterly",
    "lotsa_m1_yearly",
    # LOTSA datasets, group 3
    "lotsa_buildings_900k",
    "lotsa_era5_2015",
    "lotsa_era5_2016",
    "lotsa_era5_2017",
    "lotsa_era5_2018",
    "lotsa_uber_tlc_hourly",
    "lotsa_pedestrian_counts",
    "lotsa_solar_power",
    "lotsa_wind_farms_with_missing",
    # Datasets found only in the Chronos training corpus
    "chronos_training_corpus_kernel_synth_1m",
    "chronos_dominick",
    "chronos_ercot",
    "chronos_exchange_rate",
    "chronos_mexico_city_bikes",
    "chronos_ushcn_daily",

    # Additional LOTSA datasets (cloud ops, energy, traffic, retail, epidemiology)
    "lotsa_BEIJING_SUBWAY_30MIN",
    "lotsa_HZMETRO",
    "lotsa_LOS_LOOP",
    "lotsa_Q-TRAFFIC",
    "lotsa_SHMETRO",
    "lotsa_azure_vm_traces_2017",
    "lotsa_bdg-2_bear",
    "lotsa_bdg-2_fox",
    "lotsa_bdg-2_panther",
    "lotsa_bdg-2_rat",
    "lotsa_beijing_air_quality",
    "lotsa_bitcoin_with_missing",
    "lotsa_borealis",
    "lotsa_borg_cluster_data_2011",
    "lotsa_bull",
    "lotsa_cdc_fluview_ilinet",
    "lotsa_cdc_fluview_who_nrevss",
    "lotsa_china_air_quality",
    "lotsa_cif_2016_12",
    "lotsa_cif_2016_6",
    "lotsa_cockatoo",
    "lotsa_covid19_energy",
    "lotsa_covid_mobility",
    "lotsa_elecdemand",
    "lotsa_elf",
    "lotsa_favorita_transactions",
    "lotsa_gfc12_load",
    "lotsa_gfc14_load",
    "lotsa_gfc17_load",
    "lotsa_godaddy",
    "lotsa_hog",
    "lotsa_ideal",
    "lotsa_kaggle_web_traffic_weekly",
    "lotsa_kdd2022",
    "lotsa_largest_2017",
    "lotsa_largest_2018",
    "lotsa_largest_2019",
    "lotsa_largest_2020",
    "lotsa_largest_2021",
    "lotsa_lcl",
    "lotsa_monash_m3_other",
    "lotsa_monash_m3_quarterly",
    "lotsa_monash_m3_yearly",
    "lotsa_nn5_weekly",
    "lotsa_pdb",
    "lotsa_project_tycho",
    "lotsa_residential_load_power",
    "lotsa_residential_pv_power",
    "lotsa_rideshare_with_missing",
    "lotsa_sceaux",
    "lotsa_smart",
    "lotsa_spain",
    "lotsa_subseasonal_precip",
    "lotsa_sunspot_with_missing",
    "lotsa_taxi_30min",
    "lotsa_traffic_weekly",
    "lotsa_uber_tlc_daily",
    "lotsa_vehicle_trips_with_missing",
    "lotsa_wiki-rolling_nips",
    "gifteval",
]


class GiftEvalDatasetBuilder(LOTSADatasetBuilder):
    """
    Dataset builder for the pre-training pools.
    
    Every pool is stored in HuggingFace Arrow format under ``storage_path``
    (``$PRETRAIN_DATA`` in configs/data/*.yaml) and loads through the same
    path as the LOTSA datasets.
    
    The builder:

    - accepts the names in ``GIFT_EVAL_DATASETS``.
    - maps slash notation such as "electricity/H" to nested directories.
    - refuses the GIFT-Eval benchmark root, through ``amets.data.pool_guard``.
    
    Example:
        builder = GiftEvalDatasetBuilder(
            datasets=["electricity/H", "m4_yearly"],
            weight_map={"electricity/H": 2.0, "m4_yearly": 1.0}
        )
        dataset = builder.load_dataset(transform_map)
    """
    
    dataset_list = GIFT_EVAL_DATASETS
    dataset_type_map = defaultdict(lambda: TimeSeriesDataset)
    dataset_load_func_map = defaultdict(lambda: partial(TimeSeriesDataset))
    uniform = False
    
    def build_dataset(self, dataset: str):
        """
        Raise NotImplementedError.

        The pools are already stored in HuggingFace Arrow format. This method
        exists only to satisfy the abstract base class.

        Args:
            dataset: Dataset name.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "build_dataset() is not needed for GIFT-Eval datasets. "
            "GIFT-Eval data is already in HuggingFace Arrow format. "
            "Use load_dataset() instead."
        )
    
    def __init__(
        self,
        datasets: list[str],
        weight_map: dict[str, float] | None = None,
        storage_path: Path | str = "gift-eval-data",
        **kwargs
    ):
        """
        Initialize the builder.

        Args:
            datasets: Dataset names in slash notation, such as "electricity/H".
            weight_map: Optional mapping of dataset names to weights.
            storage_path: Root of the pre-training pools. Configs pass $PRETRAIN_DATA.
            **kwargs: Additional arguments passed to LOTSADatasetBuilder.
        """
        storage_path = Path(storage_path)
        
        # Validate dataset names.
        invalid_datasets = set(datasets) - set(self.dataset_list)
        if invalid_datasets:
            raise ValueError(
                f"Invalid datasets: {invalid_datasets}\n"
                f"Must be one of: {self.dataset_list}"
            )
        
        super().__init__(
            datasets=datasets,
            weight_map=weight_map,
            storage_path=storage_path,
            **kwargs
        )
    
    def _convert_dataset_name(self, dataset: str) -> str:
        """
        Convert a dataset name to its directory path.

        Slash notation such as "electricity/H" maps directly to the nested
        directory electricity/H/, so the name is returned unchanged.

        Args:
            dataset: Dataset name in slash notation.

        Returns:
            Directory path, equal to the input.
        """
        return dataset
    
    def load_dataset(self, transform_map):
        """
        Load the requested datasets.

        Dataset names map to nested directories under storage_path.
        amets.data.pool_guard.check_training_root refuses benchmark data first.

        Args:
            transform_map: Mapping of dataset names or types to transform functions.

        Returns:
            A Dataset, or a ConcatDataset for several datasets.
        """
        from datasets import load_from_disk
        from torch.utils.data import ConcatDataset

        from amets.data.pool_guard import check_training_root
        check_training_root(self.storage_path, self.datasets)

        datasets = []
        for dataset, weight in zip(self.datasets, self.weights):
            dir_name = self._convert_dataset_name(dataset)
            dataset_path = self.storage_path / dir_name
            
            # Check that the dataset exists.
            if not dataset_path.exists():
                raise FileNotFoundError(
                    f"Dataset not found: {dataset_path}\n"
                    f"Looking for dataset '{dataset}' (directory: '{dir_name}')"
                )
            
            # Load the HuggingFace dataset.
            from amets.data.indexer import HuggingFaceDatasetIndexer
            hf_dataset = load_from_disk(str(dataset_path))
            indexer = HuggingFaceDatasetIndexer(hf_dataset, uniform=self.uniform)
            
            transform = self._get_transform(transform_map, dataset)
            
            ts_dataset = self.dataset_load_func_map[dataset](
                indexer,
                transform,
                sample_time_series=self.sample_time_series,
                dataset_weight=weight,
                name=dataset,
            )
            datasets.append(ts_dataset)
        
        return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    
    @property
    def train_dataset(self):
        """
        Training dataset loaded with an empty transform_map.

        Training code calls load_dataset() with the real transforms instead.

        Returns:
            The loaded dataset.
        """
        return self.load_dataset(transform_map={})
