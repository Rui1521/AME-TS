import numpy as np
import torch
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from typing import Dict, Optional
import json
from sklearn.decomposition import PCA
from umap import UMAP


def visualize_adaptive_multiregime(
    embeddings_dir='router_embeddings',
    model_type="moe",
    dataset='m4_yearly',
    layer=2,
    n_samples=None,
    method='tsne',  # 'tsne', 'pca', or 'umap'
    balance_classes=True,  # balance the class distribution
    save_path=None,
    show_encoder_plot=True,  # show the encoder representation plot
    verbose=False,
    vis_features=None,  # descriptors to visualize, for example ['seasonality', 'trend']
    n_neighbors = 30,
    min_dist = 0.01
):
    """
    Plot router and encoder embeddings side by side.

    The left plot shows router logits colored by descriptor labels. The right plot
    shows encoder representations colored by expert assignment.
    
    Args:
        embeddings_dir: Directory containing router embeddings
        model_type: 'amets' for a regime-guided model, 'moe' for a standard MoE baseline
        dataset: Dataset name
        layer: Encoder layer index, starting at 0
        n_samples: Number of samples to use (None = all)
        method: Dimensionality reduction method: 'tsne', 'pca', or 'umap'.
        balance_classes: If True, balance samples across classes
        save_path: Path to save figure
        show_encoder_plot: If True, show encoder representation plot alongside router logits
        vis_features: List of descriptor names to visualize, for example ['seasonality', 'trend'].
                     If None, uses every descriptor whose threshold is above 0.
                     Valid options: 'forecastability', 'seasonality', 'trend', 'sparsity'
        verbose: If True, print detailed information
        n_neighbors: UMAP n_neighbors.
        min_dist: UMAP min_dist.
    """
    
    # Load embeddings
    embed_path = os.path.join(embeddings_dir, f'{model_type}_{dataset}')
    # Fall back to the amets_ directory for regime predictions when the model's own
    # directory has none.
    if os.path.exists(os.path.join(embed_path, 'regime_predictions.pt')):
        regime_preds = torch.load(os.path.join(embed_path, 'regime_predictions.pt'), weights_only=False)
    else:
        ame_path = os.path.join(embeddings_dir, f'amets_{dataset}')
        regime_preds = torch.load(os.path.join(ame_path, 'regime_predictions.pt'), weights_only=False)

    if model_type == "moe":
        moe_path = os.path.join(embeddings_dir, f'moe_{dataset}')
        moe_logits = torch.load(os.path.join(moe_path, f'layer_{layer}_router_logits.pt'), weights_only=False)
        router_embeddings = moe_logits.reshape(moe_logits.shape[0], -1).numpy()

        # Load encoder representations and expert assignments for encoder plot
        if show_encoder_plot:
            encoder_reps = torch.load(os.path.join(moe_path, f'layer_{layer}_encoder_reps.pt'), weights_only=False)
            encoder_embeddings = encoder_reps.reshape(encoder_reps.shape[0], -1).numpy()
            expert_assignments = torch.load(os.path.join(moe_path, f'layer_{layer}_expert_assignments.pt'), weights_only=False)
            # Take the top-1 expert from a 3D [N, patch_num, top_k] or 2D [N, top_k] array
            if expert_assignments.ndim == 3:
                expert_labels = expert_assignments[:, 0, 0].numpy()
            elif expert_assignments.ndim == 2:
                expert_labels = expert_assignments[:, 0].numpy()
            else:
                expert_labels = expert_assignments.numpy()
    elif model_type == "amets":
        logits = torch.load(os.path.join(embed_path, f'layer_{layer}_router_logits.pt'), weights_only=False)
        router_embeddings = logits.reshape(logits.shape[0], -1).numpy()

        # Load encoder representations and expert assignments for encoder plot
        if show_encoder_plot:
            encoder_reps = torch.load(os.path.join(embed_path, f'layer_{layer}_encoder_reps.pt'), weights_only=False)
            encoder_embeddings = encoder_reps.reshape(encoder_reps.shape[0], -1).numpy()
            expert_assignments = torch.load(os.path.join(embed_path, f'layer_{layer}_expert_assignments.pt'), weights_only=False)
            # Take the top-1 expert
            if expert_assignments.ndim == 3:
                expert_labels = expert_assignments[:, 0, 0].numpy()  # [N, patch_num, top_k]
            elif expert_assignments.ndim == 2:
                expert_labels = expert_assignments[:, 0].numpy()  # [N, top_k] (AME-TS flat format)
            else:
                expert_labels = expert_assignments.numpy()
    else:
        raise ValueError(f"model_type {model_type} not supported")
    
    # regime_preds is per sample but router embeddings are per token, so expand
    # regime_preds to the length of router_embeddings.
    n_router = router_embeddings.shape[0]
    n_regime = regime_preds.shape[0]
    if n_regime < n_router:
        if n_router % n_regime == 0:
            tokens_per_sample = n_router // n_regime
            regime_preds = regime_preds.repeat_interleave(tokens_per_sample, dim=0)
        else:
            # Tokens per sample vary. Approximate by repeating each prediction to
            # fill the gap, then truncate to the exact size.
            repeat_factor = (n_router // n_regime) + 1
            regime_preds = regime_preds.repeat_interleave(repeat_factor, dim=0)[:n_router]
    elif n_regime > n_router:
        regime_preds = regime_preds[:n_router]

    # Keep copies of the unsampled embeddings
    router_embeddings_orig = router_embeddings.copy()
    if show_encoder_plot:
        encoder_embeddings_orig = encoder_embeddings.copy()

    # Load thresholds from the first feature cache found. Fall back to medians.
    dataset_key = "_".join(dataset.split("/"))
    thresholds = None
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for base in [
        os.environ.get('FEATURE_CACHE', ''),
        'feature_cache',
        '../feature_cache',
        os.path.join(_repo_root, 'feature_cache'),
    ]:
        if not base:
            continue
        threshold_path = os.path.join(base, dataset_key, 'thresholds.json')
        if os.path.exists(threshold_path):
            with open(threshold_path, 'r') as f:
                thresholds = json.load(f)
            break

    regime_names = ['forecastability', 'seasonality', 'trend','sparsity']

    if thresholds is None:
        # Without cached thresholds, for example in multi-dataset mode, use the
        # median of the descriptor predictions.
        thresholds = {}
        for idx, name in enumerate(regime_names):
            if regime_preds.ndim == 3:
                vals = regime_preds[:, :, idx].mean(dim=1).numpy()
            else:
                vals = regime_preds[:, idx].numpy()
            thresholds[name] = float(np.median(vals))
        if verbose:
            print(f"Using median-based thresholds: {thresholds}")
    
    # Identify active features
    if vis_features is not None:
        # Use the descriptors the caller specified
        active_features = [f for f in vis_features if f in regime_names]
        active_indices = [regime_names.index(f) for f in active_features]
        if verbose:
            print(f"Using user-specified features: {active_features}")
    else:
        # Use every descriptor whose threshold is above 0
        active_features = []
        active_indices = []
        for idx, name in enumerate(regime_names):
            if thresholds[name] > 0:
                active_features.append(name)
                active_indices.append(idx)
    
    if verbose:
        print(f"Dataset: {dataset}, Model: {model_type}, Layer: {layer}")
        print(f"Active features: {active_features}")
        print(f"Thresholds: {[thresholds[f] for f in active_features]}")
    
    # Build multi-descriptor labels from the active descriptors only
    n_active = len(active_features)
    
    if n_active == 0:
        print("Warning: no active descriptors, so no labels can be built.")
        return None
    
    # Build a binary label for each active descriptor
    feature_labels = []
    for idx in active_indices:
        if regime_preds.ndim == 3:
            # (N, patch_num, num_regimes) -> average across patches
            regime_values = regime_preds[:, :, idx].mean(dim=1).numpy()
        else:
            # (N, num_regimes) -> already per-sample
            regime_values = regime_preds[:, idx].numpy()
        threshold = thresholds[regime_names[idx]]
        binary_labels = (regime_values > threshold).astype(int)
        feature_labels.append(binary_labels)
    
    # Combine into multi-class labels
    labels = np.zeros(len(feature_labels[0]), dtype=int)
    for i, feat_labels in enumerate(feature_labels):
        labels += feat_labels * (2 ** i)
    
    # Create label names
    n_classes = 2 ** n_active
    label_names = []
    for i in range(n_classes):
        binary = format(i, f'0{n_active}b')
        name_parts = []
        for j, (feat_name, bit) in enumerate(zip(active_features, binary)):
            level = 'High' if bit == '1' else 'Low'
            name_parts.append(f"{level} {feat_name.capitalize()}")
        label_names.append(', '.join(name_parts))
    
    # Print the label distribution before sampling
    if verbose:
        print(f"\nLabel distribution before sampling ({n_classes} classes):")
    class_counts = {}
    for i in range(n_classes):
        count = np.sum(labels == i)
        class_counts[i] = count
        if verbose:
            if count > 0:
                print(f"  {label_names[i]}: {count}")
        
    # Balanced sampling
    if balance_classes and n_samples:
        # Find minimum class size
        existing_classes = [i for i in range(n_classes) if class_counts[i] > 0]
        min_class_size = min([class_counts[i] for i in existing_classes])
        
        # Calculate samples per class
        samples_per_class = min(min_class_size, n_samples // len(existing_classes))
        if verbose:
            print(f"\nBalanced sampling: {samples_per_class} samples per class")
        
        # Sample from each class
        balanced_indices = []
        for class_idx in existing_classes:
            class_mask = (labels == class_idx)
            class_indices = np.where(class_mask)[0]
            sampled = np.random.choice(class_indices, size=samples_per_class, replace=False)
            balanced_indices.extend(sampled)
        
        balanced_indices = np.array(balanced_indices)
        np.random.shuffle(balanced_indices)
        
        # Apply balanced sampling to all embeddings
        router_embeddings = router_embeddings[balanced_indices]
        if show_encoder_plot:
            encoder_embeddings = encoder_embeddings[balanced_indices]
            expert_labels = expert_labels[balanced_indices]
        labels = labels[balanced_indices]
        regime_preds = regime_preds[balanced_indices]
        if verbose:
            print(f"Total samples after balancing: {len(router_embeddings)}")
            print(f"Balanced distribution:")
            for i in existing_classes:
                count = np.sum(labels == i)
                print(f"  {label_names[i]}: {count}")
        
    elif n_samples and not balance_classes:
        # Uniform random sampling without class balancing.
        # Cap n_samples at the number of available samples.
        actual_n_samples = min(n_samples, len(router_embeddings))
        if actual_n_samples < n_samples:
            print(f"\nWarning: Requested {n_samples} samples but only {len(router_embeddings)} available.")
            print(f"Using all {actual_n_samples} samples.")
        
        indices = np.random.choice(len(router_embeddings), size=actual_n_samples, replace=False)
        router_embeddings = router_embeddings[indices]
        if show_encoder_plot:
            encoder_embeddings = encoder_embeddings[indices]
            expert_labels = expert_labels[indices]
        print(f"Unique labels: {np.unique(labels)}")
        print(f"Label counts: {np.bincount(labels.astype(int))}")
        labels = labels[indices]
        regime_preds = regime_preds[indices]
    # Check for NaN values in router embeddings
    if np.isnan(router_embeddings).any():
        nan_count = np.isnan(router_embeddings).sum()
        print(f"Warning: found {nan_count} NaN values in router embeddings")
        valid_mask = ~np.isnan(router_embeddings).any(axis=1)
        router_embeddings = router_embeddings[valid_mask]
        labels = labels[valid_mask]
        if show_encoder_plot:
            encoder_embeddings = encoder_embeddings[valid_mask]
            expert_labels = expert_labels[valid_mask]
        print(f"Removed {(~valid_mask).sum()} samples with NaN. Remaining: {len(router_embeddings)}")
    
    # Check for NaN values in encoder embeddings
    if show_encoder_plot and np.isnan(encoder_embeddings).any():
        nan_count = np.isnan(encoder_embeddings).sum()
        print(f"Warning: found {nan_count} NaN values in encoder embeddings")
        valid_mask = ~np.isnan(encoder_embeddings).any(axis=1)
        router_embeddings = router_embeddings[valid_mask]
        encoder_embeddings = encoder_embeddings[valid_mask]
        labels = labels[valid_mask]
        expert_labels = expert_labels[valid_mask]
        print(f"Removed {(~valid_mask).sum()} samples with NaN. Remaining: {len(router_embeddings)}")
    
    # Check for sufficient samples
    if len(router_embeddings) < 10:
        raise ValueError(f"Too few valid samples ({len(router_embeddings)}). Cannot visualize.")
    
    if verbose:
        # Apply dimensionality reduction: t-SNE, PCA, or UMAP
        print(f"\nApplying {method.upper()}...")
    
    if method.lower() == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        proj_router = reducer.fit_transform(router_embeddings)
        explained_var = reducer.explained_variance_ratio_
        if verbose:
            print(f"PCA explained variance (Router): {explained_var[0]:.3f}, {explained_var[1]:.3f} (total: {explained_var.sum():.3f})")
        dim1_label = f'PC1 ({explained_var[0]:.1%})'
        dim2_label = f'PC2 ({explained_var[1]:.1%})'
        
        if show_encoder_plot:
            reducer_enc = PCA(n_components=2, random_state=42)
            proj_encoder = reducer_enc.fit_transform(encoder_embeddings)
            explained_var_enc = reducer_enc.explained_variance_ratio_
            if verbose:
                print(f"PCA explained variance (Encoder): {explained_var_enc[0]:.3f}, {explained_var_enc[1]:.3f} (total: {explained_var_enc.sum():.3f})")
            dim1_label_enc = f'PC1 ({explained_var_enc[0]:.1%})'
            dim2_label_enc = f'PC2 ({explained_var_enc[1]:.1%})'
    elif method.lower() == 'tsne':
        perplexity = min(30, len(router_embeddings) // 4)
        reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        proj_router = reducer.fit_transform(router_embeddings)
        dim1_label = 't-SNE Dimension 1'
        dim2_label = 't-SNE Dimension 2'
        
        if show_encoder_plot:
            reducer_enc = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            proj_encoder = reducer_enc.fit_transform(encoder_embeddings)
            dim1_label_enc = 't-SNE Dimension 1'
            dim2_label_enc = 't-SNE Dimension 2'
    elif method.lower() == 'umap':
        reducer = UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors, min_dist=min_dist, metric="cosine")
        proj_router = reducer.fit_transform(router_embeddings)
        dim1_label = 'UMAP Dimension 1'
        dim2_label = 'UMAP Dimension 2'
        
        if show_encoder_plot:
            reducer_enc = UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors, min_dist=min_dist, metric="cosine")
            proj_encoder = reducer_enc.fit_transform(encoder_embeddings)
            dim1_label_enc = 'UMAP Dimension 1'
            dim2_label_enc = 'UMAP Dimension 2'

    else:
        raise ValueError(f"method must be 'tsne', 'pca', or 'umap', got {method}")
    
    # Compute Calinski-Harabasz scores when there is more than one label
    unique_labels = len(np.unique(labels))
    if unique_labels > 1:
        from sklearn.metrics import calinski_harabasz_score
        ch_router = calinski_harabasz_score(router_embeddings, labels)
        if verbose:
            print(f"Router Embeddings - Calinski-Harabasz Score: {ch_router:.3f}")
    else:
        if verbose:
            print(f"Warning: only {unique_labels} unique label(s) in router embeddings. Skipping Calinski-Harabasz score.")
        ch_router = None
    
    if show_encoder_plot:
        unique_expert_labels = len(np.unique(expert_labels))
        if unique_expert_labels > 1:
            from sklearn.metrics import calinski_harabasz_score
            ch_encoder = calinski_harabasz_score(encoder_embeddings, expert_labels)
            if verbose:
                print(f"Encoder Representations - Calinski-Harabasz Score: {ch_encoder:.3f}")
        else:
            if verbose:
                print(f"Warning: only {unique_expert_labels} unique expert label(s). Skipping Calinski-Harabasz score.")
            ch_encoder = None
    
    # Create side-by-side plots
    if show_encoder_plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(19, 8))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))
    
    colors = sns.color_palette("husl", n_classes)
    
    # Left plot: router logits colored by descriptor labels
    for i in range(n_classes):
        mask = labels == i
        if np.sum(mask) > 0:  # plot only classes that occur
            ax1.scatter(proj_router[mask, 0], proj_router[mask, 1],
                       c=[colors[i]], label=label_names[i], 
                       alpha=0.6, s=30, edgecolors='white', linewidth=0.5)
    
    ax1.set_xlabel(dim1_label, fontsize=16)
    ax1.set_ylabel(dim2_label, fontsize=16)
    ch_str = f'{ch_router:.3f}' if ch_router is not None else 'N/A'
    display_name = 'Regime-Aware' if model_type == 'amets' else 'Standard MoE'
    title_left = f'{display_name} Router Space, Layer {layer} (CH={ch_str})'
    ax1.set_title(title_left, fontsize=19, fontweight='bold')
    ax1.legend(loc='best', fontsize=15, framealpha=0.9, ncol=1 if n_classes <= 4 else 2)
    ax1.grid(True, alpha=0.3)
    
    # Right plot: encoder representations colored by expert assignment
    if show_encoder_plot:
        unique_experts = np.unique(expert_labels)
        n_experts = len(unique_experts)
        expert_colors = sns.color_palette("husl", n_experts)

        # Cap each expert at 3x the median count, with a floor of 200, so dominant
        # experts do not hide the others. Counts below the cap stay as they are.
        expert_counts = [np.sum(expert_labels == e) for e in unique_experts]
        cap = max(int(np.median(expert_counts)) * 3, 200)
        balanced_indices = []
        for expert_id in unique_experts:
            expert_idx = np.where(expert_labels == expert_id)[0]
            if len(expert_idx) > cap:
                expert_idx = np.random.choice(expert_idx, cap, replace=False)
            balanced_indices.extend(expert_idx)
        balanced_indices = np.array(balanced_indices)

        proj_encoder_balanced = proj_encoder[balanced_indices]
        expert_labels_balanced = expert_labels[balanced_indices]

        # Recompute the CH score on the capped data
        if len(np.unique(expert_labels_balanced)) > 1:
            from sklearn.metrics import calinski_harabasz_score
            ch_encoder = calinski_harabasz_score(encoder_embeddings[balanced_indices], expert_labels_balanced)

        # Map each expert_id to a color index
        expert_to_color_idx = {expert_id: idx for idx, expert_id in enumerate(unique_experts)}

        for expert_id in unique_experts:
            mask = expert_labels_balanced == expert_id
            if np.sum(mask) > 0:
                color_idx = expert_to_color_idx[expert_id]
                ax2.scatter(proj_encoder_balanced[mask, 0], proj_encoder_balanced[mask, 1],
                           c=[expert_colors[color_idx]], label=f'Expert {int(expert_id)}',
                           alpha=0.6, s=30, edgecolors='white', linewidth=0.5)
        
        ax2.set_xlabel(dim1_label_enc, fontsize=16)
        ax2.set_ylabel(dim2_label_enc, fontsize=16)
        ch_enc_str = f'{ch_encoder:.3f}' if ch_encoder is not None else 'N/A'
        title_right = f'{display_name} Encoder Space, Layer {layer} (CH={ch_enc_str})'
        ax2.set_title(title_right, fontsize=19, fontweight='bold')
        ax2.legend(loc='best', fontsize=15, framealpha=0.9)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"\nSaved to {save_path}")
    
    plt.show()
    
    result = {
        'ch_router': ch_router,
        'active_features': active_features,
        'n_classes': n_classes,
        'method': method,
        'label_distribution': {label_names[i]: int(np.sum(labels == i)) 
                              for i in range(n_classes) if np.sum(labels == i) > 0}
    }
    
    if show_encoder_plot:
        result['ch_encoder'] = ch_encoder
        result['n_experts'] = n_experts
    
    return result



def generate_comprehensive_comparison_table(
    ame_results_path='results/ame/all_results.csv',
    baseline_paths=None,
    improvement_baseline=None,
    metrics=None,
    output_format='markdown',
    save_path=None,
    exclude_datasets=None,
    verbose=True
):
    """
    Build a table that compares AME-TS results with baseline results.

    Each row is a metric and each column is a model, AME-TS or a baseline. Extra
    columns hold per-task win counts and the improvement over a chosen baseline.

    Args:
        ame_results_path: Path to the AME-TS results CSV. The function keeps the row
            with the lowest metrics[0] per dataset, so pass a CSV that already holds
            one selected config per task.
        baseline_paths: Dict of {model_name: csv_path} for baselines.
            Example: {'BaselineA': 'path/to/baseline_a/all_results.csv',
                      'BaselineB': 'path/to/baseline_b/all_results.csv'}
        improvement_baseline: Name of baseline to compute improvement against.
            Must be a key in baseline_paths. If None, uses the first baseline.
        metrics: List of metric column names. If None, uses MSE, MAE, MASE, sMAPE and RMSE.
        output_format: 'markdown', 'latex', or 'dataframe'
        save_path: Optional path to save the table
        exclude_datasets: List of dataset base names that the improvement column
            leaves out. They still count as tasks. Default: none.
        verbose: If True, print detailed statistics
    
    Returns:
        pd.DataFrame: Comparison table with one row per metric.

    Example:
        >>> table = generate_comprehensive_comparison_table(
        ...     ame_results_path='results/ame/all_results.csv',
        ...     baseline_paths={
        ...         'BaselineA': 'path/to/baseline_a/all_results.csv',
        ...         'BaselineB': 'path/to/baseline_b/all_results.csv',
        ...     },
        ...     improvement_baseline='BaselineA',
        ... )
    """
    import pandas as pd
    import numpy as np
    from scipy import stats

    if baseline_paths is None:
        baseline_paths = {}
    if exclude_datasets is None:
        exclude_datasets = []
    if improvement_baseline is None and baseline_paths:
        improvement_baseline = list(baseline_paths.keys())[0]

    # Default metrics
    if metrics is None:
        metrics = [
            'eval_metrics/MSE[mean]',
            'eval_metrics/MAE[0.5]',
            'eval_metrics/MASE[0.5]',
            'eval_metrics/sMAPE[0.5]',
            'eval_metrics/RMSE[mean]'
        ]

    # Load AME-TS results and keep the lowest-metric row per dataset
    ame_df = pd.read_csv(ame_results_path)
    ame_best = ame_df.loc[ame_df.groupby('dataset')[metrics[0]].idxmin()]
    if verbose:
        print(f"AME-TS: {len(ame_df)} rows -> {len(ame_best)} datasets (best per context)")

    # Load baselines and keep the lowest-metric row per dataset
    baseline_dfs = {}
    for name, path in baseline_paths.items():
        df = pd.read_csv(path)
        best = df.loc[df.groupby('dataset')[metrics[0]].idxmin()]
        baseline_dfs[name] = best
        if verbose:
            print(f"{name}: {len(df)} rows -> {len(best)} datasets")

    # Build the results table with one row per metric and one column per model
    results = []

    for metric in metrics:
        metric_name = metric.split('/')[-1].replace('[mean]', '').replace('[0.5]', '')
        row = {'Metric': metric_name}

        # AME-TS metric values per dataset
        ame_vals = ame_best.set_index('dataset')[metric]

        # Baseline metric values per dataset
        baseline_vals = {}
        for name, best_df in baseline_dfs.items():
            baseline_vals[name] = best_df.set_index('dataset')[metric]

        # Find the datasets common to AME-TS and all baselines
        common = set(ame_vals.dropna().index)
        for name, vals in baseline_vals.items():
            common = common & set(vals.dropna().index)
        common = sorted(common)

        if len(common) == 0:
            if verbose:
                print(f"Warning: No common datasets for metric {metric_name}")
            continue

        total_tasks = len(common)

        # Load seasonal-naive results for the GIFT-Eval normalization.
        # The CSV lives in the gift-eval checkout ($GIFT_EVAL_REPO). Set
        # SEASONAL_NAIVE_RESULTS to use another copy. Without it, the table
        # reports raw metric means and skips the normalized score.
        _repo = os.environ.get('GIFT_EVAL_REPO', os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'gift-eval'))
        sn_path = os.environ.get(
            'SEASONAL_NAIVE_RESULTS',
            os.path.join(_repo, 'results', 'seasonal_naive', 'all_results.csv'),
        )
        sn_vals = None
        if os.path.exists(sn_path):
            sn_df = pd.read_csv(sn_path)
            sn_best = sn_df.loc[sn_df.groupby('dataset')[metrics[0]].idxmin()] if len(sn_df) > len(sn_df['dataset'].unique()) else sn_df
            sn_vals = sn_best.set_index('dataset')[metric]

        # Geometric mean of model / seasonal_naive, the GIFT-Eval scoring rule
        def geo_mean_score(model_vals, common_ds, sn_vals):
            if sn_vals is None:
                return None
            ratios = []
            for ds in common_ds:
                if ds in sn_vals.index and sn_vals[ds] > 0:
                    ratios.append(model_vals[ds] / sn_vals[ds])
            if ratios:
                return np.exp(np.mean(np.log(np.clip(ratios, 1e-10, None))))
            return None

        # Model scores: geometric mean normalized by seasonal naive
        ame_common = ame_vals[common]
        ame_geo = geo_mean_score(ame_vals, common, sn_vals)
        row['AME-TS'] = f"{ame_geo:.4f}" if ame_geo else f"{ame_common.mean():.4f}"

        for name in baseline_paths:
            bl_common = baseline_vals[name][common]
            bl_geo = geo_mean_score(baseline_vals[name], common, sn_vals)
            row[name] = f"{bl_geo:.4f}" if bl_geo else f"{bl_common.mean():.4f}"

        # Per-task win count of AME-TS against each baseline
        for name in baseline_paths:
            bl_common = baseline_vals[name][common]
            wins = (ame_vals[common].values < bl_common.values).sum()
            row[f'AME-TS_wins_vs_{name}'] = f"{wins}/{total_tasks}"

        # Improvement over the chosen baseline, leaving out exclude_datasets
        if improvement_baseline and improvement_baseline in baseline_vals:
            bl_imp = baseline_vals[improvement_baseline]

            # Leave out the datasets in exclude_datasets
            filtered_common = [d for d in common
                               if not any(d.startswith(e) for e in exclude_datasets)]
            n_excluded = len(common) - len(filtered_common)

            # Improvement in the geometric-mean score
            ame_geo_filt = geo_mean_score(ame_vals, filtered_common, sn_vals)
            bl_geo_filt = geo_mean_score(bl_imp, filtered_common, sn_vals)

            if ame_geo_filt and bl_geo_filt and bl_geo_filt > 0:
                geo_improve = (bl_geo_filt - ame_geo_filt) / bl_geo_filt * 100
            else:
                # Without seasonal naive, use the mean per-task improvement
                ame_f = ame_vals[filtered_common].values
                bl_f = bl_imp[filtered_common].values
                geo_improve = ((bl_f - ame_f) / bl_f * 100).mean()

            # Paired t-test p-value on the same datasets
            ame_f = ame_vals[filtered_common].values
            bl_f = bl_imp[filtered_common].values
            if len(ame_f) > 2:
                _, pvalue = stats.ttest_rel(bl_f, ame_f)
            else:
                pvalue = 1.0

            row[f'Improve_vs_{improvement_baseline}'] = f"{geo_improve:+.1f}%"
            row[f'p-value'] = f"{pvalue:.4f}" if pvalue >= 0.0001 else "<0.0001"

        results.append(row)

    results_df = pd.DataFrame(results)

    if verbose:
        print(f"\nComparison across {total_tasks} tasks:")
        print(results_df.to_string(index=False))

    # Format output
    if output_format == 'markdown':
        output = results_df.to_markdown(index=False)
    elif output_format == 'latex':
        output = results_df.to_latex(index=False)
    elif output_format == 'dataframe':
        output = results_df
    else:
        raise ValueError(f"Unknown output format: {output_format}")

    if save_path:
        if output_format == 'dataframe':
            results_df.to_csv(save_path, index=False)
        else:
            with open(save_path, 'w') as f:
                f.write(output)
        if verbose:
            print(f"\nSaved to: {save_path}")

    return results_df

