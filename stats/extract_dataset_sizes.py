#!/usr/bin/env python3
"""
Extract dataset sizes (number of time series) for all GIFT-Eval datasets.

The script loads each dataset and counts the time series in its test set. The
counts can weight per-dataset results.

Usage:
    python stats/extract_dataset_sizes.py
    
Output:
    dataset_sizes.json in the working directory, with the test set size of each dataset
"""

import json
import sys
import os
from tqdm import tqdm

# Put the repository root on sys.path and make gift_eval importable from a local checkout.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from amets.common.env import add_gift_eval_to_path, dataset_properties_path  # noqa: E402

add_gift_eval_to_path()

from gift_eval.data import Dataset as GiftDataset


def get_dataset_size(dataset_name, term='short'):
    """
    Count the time series in a dataset's test set.
    
    For datasets with frequency variants, tries common frequency suffixes.
    
    Args:
        dataset_name: Name of dataset (e.g., 'm4_yearly', 'electricity')
        term: Dataset term (default: 'short')
        
    Returns:
        Dictionary with dataset size info
    """
    # Map dataset_properties.json names to actual directory names
    name_mapping = {
        'loop_seattle': 'LOOP_SEATTLE',
        'sz_taxi': 'SZ_TAXI',
        'm_dense': 'M_DENSE',
        'saugeen': 'saugeenday',  # the data directory is named saugeenday
        'temperature_rain': 'temperature_rain_with_missing',
        'kdd_cup_2018': 'kdd_cup_2018_with_missing',
        'car_parts': 'car_parts_with_missing',
    }
    
    # Apply name mapping if needed
    actual_name = name_mapping.get(dataset_name, dataset_name)
    
    # Try loading dataset directly first
    try:
        dataset = GiftDataset(name=actual_name, term=term, to_univariate=False)
        n_test = len(list(dataset.test_data.input))
        
        return {
            'n_test': n_test,
            'success': True,
            'variant_used': actual_name
        }
    except Exception as e:
        # If direct load fails, try with frequency variants
        freq_variants = ['H', 'D', 'W', 'M', '15T', '10T', '5T']
        
        for freq in freq_variants:
            try:
                variant_name = f"{actual_name}/{freq}"
                dataset = GiftDataset(name=variant_name, term=term, to_univariate=False)
                n_test = len(list(dataset.test_data.input))
                
                return {
                    'n_test': n_test,
                    'success': True,
                    'variant_used': variant_name
                }
            except:
                continue
        
        # All attempts failed
        return {
            'n_test': None,
            'success': False,
            'error': str(e),
            'variant_used': None
        }


def main():
    print("Extracting dataset sizes from GIFT-Eval...")
    print("=" * 60)
    
    # Load dataset properties to get list of datasets
    props_path = dataset_properties_path()
    with open(props_path) as f:
        props = json.load(f)
    
    print(f"Found {len(props)} datasets in dataset_properties.json\n")
    
    dataset_sizes = {}
    failed_datasets = []
    
    for dataset_name in tqdm(sorted(props.keys()), desc="Processing datasets"):
        result = get_dataset_size(dataset_name, term='short')
        
        if result['success']:
            dataset_sizes[dataset_name] = {
                'n_test': result['n_test'],
                'num_variates': props[dataset_name]['num_variates'],
                'domain': props[dataset_name]['domain']
            }
            print(f"OK   {dataset_name:30s}: {result['n_test']:6d} test series")
        else:
            failed_datasets.append((dataset_name, result['error']))
            print(f"FAIL {dataset_name:30s}: {result['error']}")
    
    # Save to JSON
    output_path = 'dataset_sizes.json'
    with open(output_path, 'w') as f:
        json.dump(dataset_sizes, f, indent=2, sort_keys=True)
    
    print("\n" + "=" * 60)
    print(f"Saved dataset sizes to {output_path}")
    print(f"  - Successfully processed: {len(dataset_sizes)} datasets")
    print(f"  - Failed: {len(failed_datasets)} datasets")
    
    if failed_datasets:
        print("\nFailed datasets:")
        for name, error in failed_datasets:
            print(f"  - {name}: {error}")
    
    # Print summary statistics
    test_sizes = [info['n_test'] for info in dataset_sizes.values()]
    print(f"\nTest set size statistics:")
    print(f"  - Min: {min(test_sizes):,} series")
    print(f"  - Max: {max(test_sizes):,} series")
    print(f"  - Mean: {sum(test_sizes) / len(test_sizes):,.0f} series")
    print(f"  - Median: {sorted(test_sizes)[len(test_sizes)//2]:,} series")
    
    # Show the largest datasets
    print(f"\nTop 10 largest datasets (by test set size):")
    sorted_datasets = sorted(dataset_sizes.items(), key=lambda x: x[1]['n_test'], reverse=True)
    for i, (name, info) in enumerate(sorted_datasets[:10], 1):
        print(f"  {i:2d}. {name:30s}: {info['n_test']:6,} series")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
