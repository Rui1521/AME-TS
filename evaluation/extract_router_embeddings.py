"""
Extract router embeddings from a trained AME-TS checkpoint.

The script extracts router logits, routing probabilities, expert assignments and
encoder representations layer by layer for interpretability analysis. It saves
them to <output_dir>/<prefix>_<dataset>/, where the prefix is --save_prefix or
"amets".

Usage (from the repository root):
    python evaluation/extract_router_embeddings.py --dataset m4_yearly \
        --checkpoint /path/to/checkpoint.ckpt --max_context 512

    # A standard MoE baseline checkpoint, saved under the "moe" prefix
    python evaluation/extract_router_embeddings.py --dataset m4_yearly --save_prefix moe \
        --checkpoint /path/to/moe_checkpoint.ckpt
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List
import argparse

# Put the repository root on sys.path.
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)


def load_amets_model(
    checkpoint_path: str,
    device: str = 'cuda'
):
    """Load a trained AME-TS model from a Lightning checkpoint."""
    from amets.model.pretrain import AMETSPretrain

    model = AMETSPretrain.load_from_checkpoint(checkpoint_path, map_location=device)
    model.to(device)
    model.eval()

    print(f"Loaded AME-TS model from {checkpoint_path}")
    print(f"  d_model: {model.module.d_model}")
    print(f"  num_layers: {model.module.num_layers}")
    print(f"  num_experts: {model.module.num_experts}")
    print(f"  routing_mode: {model.module.routing_mode}")
    return model


def extract_amets_router_embeddings(
    model,
    dataset_name: str,
    term: str = 'short',
    max_context: int = 512,
    max_samples: int = 5000,
    device: str = 'cuda'
) -> Dict[str, Any]:
    """
    Extract router embeddings from an AME-TS model.

    Runs each GIFT-Eval test input through the model and records per-layer
    routing information.

    Args:
        model: Loaded AMETSPretrain model
        dataset_name: GIFT-eval dataset name (e.g., 'm4_yearly')
        term: Dataset term ('short', 'medium', 'long')
        max_context: Maximum context length
        max_samples: Maximum number of samples to process
        device: Device to run on

    Returns:
        Dictionary with routing embeddings per layer
    """
    from amets.common.env import add_gift_eval_to_path

    add_gift_eval_to_path()
    from gift_eval.data import Dataset as GiftDataset

    print(f"Extracting AME-TS router embeddings for {dataset_name}/{term}...")

    # Load dataset
    ds = GiftDataset(name=dataset_name, term=term, to_univariate=False)
    pred_len = ds.prediction_length

    # Get encoder layers
    module = model.module
    encoder = module.encoder
    encoder_layers = encoder.layers
    n_layers = len(encoder_layers)

    # Determine patch size
    patch_sizes = list(module.patch_sizes)
    patch_size = patch_sizes[len(patch_sizes) // 2]  # Use middle patch size

    print(f"  Encoder layers: {n_layers}")
    print(f"  Patch size: {patch_size}")
    print(f"  Prediction length: {pred_len}")
    print(f"  Max context: {max_context}")

    # Create storage
    embeddings = {'regime_predictions': []}
    for i in range(n_layers):
        embeddings[f'layer_{i}'] = {
            'router_logits': [],
            'routing_probs': [],
            'expert_assignments': [],
            'encoder_reps': [],
            'expert_usage': [],
        }

    sample_count = 0

    with torch.no_grad():
        for entry in ds.test_data:
            if sample_count >= max_samples:
                break

            # test_data yields (input_dict, label_dict) tuples
            if isinstance(entry, tuple):
                input_entry, label_entry = entry
            else:
                input_entry = entry

            target = np.array(input_entry['target'], dtype=np.float32)
            if target.ndim == 1:
                target = target.reshape(1, -1)  # (1, time)

            n_variates = target.shape[0]
            context_len = min(target.shape[1], max_context)

            if context_len < patch_size * 2:
                continue

            # input_entry['target'] is already the context (no prediction window)
            context = target[:, -context_len:]  # (n_variates, context_len)

            # Process each variate independently, as in AME-TS training
            for v in range(n_variates):
                variate_ctx = context[v]  # (context_len,)

                # Compute regime probs
                ctx_tensor = torch.from_numpy(variate_ctx).float().to(device)
                ctx_tensor = torch.nan_to_num(ctx_tensor, nan=0.0)

                # Get regime predictions from the model's predictor
                regime_predictor = module.regime_predictor
                probs = regime_predictor.get_probabilities(ctx_tensor.unsqueeze(0))  # (1, 3 or 4)

                # Append analytic sparsity if the predictor outputs fewer than num_regimes
                if probs.shape[-1] < module.num_regimes:
                    sparsity = (ctx_tensor == 0).float().mean().unsqueeze(0).unsqueeze(0).to(device)
                    probs = torch.cat([probs, sparsity], dim=-1)

                embeddings['regime_predictions'].append(probs.cpu())

                # Patch the context
                num_patches = context_len // patch_size
                if num_patches < 2:
                    continue

                max_patch_size = max(patch_sizes)

                # Create target tensor: (num_patches, max_patch_size)
                target_flat = torch.zeros(num_patches, max_patch_size, device=device)
                target_flat[:, :patch_size] = ctx_tensor[:num_patches * patch_size].reshape(num_patches, patch_size)

                # Create masks and IDs
                # observed_mask marks only real data within patch_size, not padding
                observed_mask = torch.zeros(num_patches, max_patch_size, dtype=torch.bool, device=device)
                observed_mask[:, :patch_size] = True
                sample_id = torch.zeros(num_patches, dtype=torch.long, device=device)
                variate_id_tensor = torch.zeros(num_patches, dtype=torch.long, device=device)
                time_id = torch.arange(num_patches, dtype=torch.long, device=device)
                prediction_mask = torch.zeros(num_patches, dtype=torch.bool, device=device)
                patch_size_tensor = torch.full((num_patches,), patch_size, dtype=torch.long, device=device)

                # Expand regime_probs to num_patches
                regime_probs_expanded = probs.expand(num_patches, -1)  # (num_patches, num_regimes)

                # Register hooks to capture router inputs at each layer
                hook_data = {}

                def make_hook(layer_idx):
                    def hook_fn(ffn_module, args, kwargs):
                        x_input = args[0] if args else kwargs.get('x')
                        if x_input is not None:
                            x_flat = x_input.view(-1, x_input.shape[-1])
                            router_logits = ffn_module.router(x_flat)
                            routing_probs_layer = F.softmax(router_logits, dim=-1)
                            top_k_weights, top_k_indices = torch.topk(
                                routing_probs_layer, ffn_module.num_experts_per_token, dim=-1
                            )
                            expert_counts = torch.zeros(ffn_module.num_experts, device=x_flat.device)
                            for e in range(ffn_module.num_experts):
                                expert_counts[e] = (top_k_indices == e).sum().float()
                            expert_usage = expert_counts / top_k_indices.numel()

                            hook_data[layer_idx] = {
                                'encoder_reps': x_flat.cpu(),
                                'router_logits': router_logits.cpu(),
                                'routing_probs': routing_probs_layer.cpu(),
                                'expert_assignments': top_k_indices.cpu(),
                                'expert_usage': expert_usage.cpu(),
                            }
                    return hook_fn

                # Register forward pre-hooks on each MoE FFN layer
                hooks = []
                for layer_idx, layer in enumerate(encoder_layers):
                    if hasattr(layer.ffn, 'router'):
                        h = layer.ffn.register_forward_pre_hook(make_hook(layer_idx), with_kwargs=True)
                        hooks.append(h)

                # Run full forward pass through the module
                try:
                    output, *_ = module(
                        target=target_flat,
                        observed_mask=observed_mask,
                        sample_id=sample_id,
                        time_id=time_id,
                        variate_id=variate_id_tensor,
                        prediction_mask=prediction_mask,
                        patch_size=patch_size_tensor,
                        regime_probs=regime_probs_expanded if module.routing_mode == 'additive' else None,
                    )
                except Exception as e:
                    # Remove hooks and skip this sample
                    for h in hooks:
                        h.remove()
                    print(f"  Warning: forward pass failed for sample {sample_count}: {e}")
                    continue

                # Remove hooks
                for h in hooks:
                    h.remove()

                # Collect hook data into embeddings
                for layer_idx, data in hook_data.items():
                    for key in ['encoder_reps', 'router_logits', 'routing_probs', 'expert_assignments', 'expert_usage']:
                        if key in data:
                            embeddings[f'layer_{layer_idx}'][key].append(data[key])

                sample_count += 1

            if sample_count % 100 == 0 and sample_count > 0:
                print(f"  Processed {sample_count} samples...")

    # Concatenate all samples
    if embeddings['regime_predictions']:
        embeddings['regime_predictions'] = torch.cat(embeddings['regime_predictions'], dim=0)

    for layer_key in embeddings:
        if layer_key == 'regime_predictions':
            continue
        for embed_key in embeddings[layer_key]:
            if embeddings[layer_key][embed_key]:
                embeddings[layer_key][embed_key] = torch.cat(
                    embeddings[layer_key][embed_key], dim=0
                )
            else:
                embeddings[layer_key][embed_key] = torch.tensor([])

    print(f"Extraction complete. Processed {sample_count} samples.")
    return embeddings


def save_embeddings(embeddings: Dict, output_dir: str, model_type: str, dataset_name: str):
    """Save extracted embeddings to disk."""
    save_dir = os.path.join(output_dir, f"{model_type}_{dataset_name}")
    os.makedirs(save_dir, exist_ok=True)

    for layer_key in embeddings:
        if layer_key == 'regime_predictions':
            if isinstance(embeddings[layer_key], torch.Tensor) and embeddings[layer_key].numel() > 0:
                save_path = os.path.join(save_dir, 'regime_predictions.pt')
                torch.save(embeddings[layer_key], save_path)
                print(f"Saved regime predictions to {save_path}")
                print(f"  Shape: {embeddings[layer_key].shape}")
        else:
            for embed_key, embed_tensor in embeddings[layer_key].items():
                if isinstance(embed_tensor, torch.Tensor) and embed_tensor.numel() > 0:
                    save_path = os.path.join(save_dir, f"{layer_key}_{embed_key}.pt")
                    torch.save(embed_tensor, save_path)
                    print(f"Saved {layer_key}/{embed_key}: {embed_tensor.shape}")


def _concatenate_embeddings(embeddings_list: List[Dict]) -> Dict:
    """Concatenate embeddings from multiple datasets into one.

    Each embedding dict has structure:
        {'regime_predictions': Tensor, 'layer_0': {'router_logits': T, 'encoder_reps': T, ...}, ...}
    """
    combined = {}

    # Get all keys from first embedding
    ref = embeddings_list[0]

    for key in ref:
        if key == 'regime_predictions':
            # Concatenate regime predictions
            tensors = [e[key] for e in embeddings_list if key in e and isinstance(e[key], torch.Tensor) and e[key].numel() > 0]
            if tensors:
                combined[key] = torch.cat(tensors, dim=0)
        else:
            # Layer dict: concatenate each sub-key
            combined[key] = {}
            for sub_key in ref[key]:
                tensors = [e[key][sub_key] for e in embeddings_list
                          if key in e and sub_key in e[key]
                          and isinstance(e[key][sub_key], torch.Tensor) and e[key][sub_key].numel() > 0]
                if tensors:
                    combined[key][sub_key] = torch.cat(tensors, dim=0)

    # Print summary
    print(f"\nCombined embeddings from {len(embeddings_list)} datasets:")
    if 'regime_predictions' in combined:
        print(f"  regime_predictions: {combined['regime_predictions'].shape}")
    for key in combined:
        if key != 'regime_predictions':
            for sub_key, t in combined[key].items():
                print(f"  {key}/{sub_key}: {t.shape}")
            break  # print only the first layer as an example

    return combined


def main():
    parser = argparse.ArgumentParser(description='Extract router embeddings from a trained AME-TS checkpoint')
    parser.add_argument('--dataset', type=str, default='m4_yearly',
                        help='Dataset name(s). Comma-separated for multi-dataset mode, e.g. "m4_monthly,lotsa_m5,hospital"')
    parser.add_argument('--term', type=str, default='short',
                        help='Dataset term: short, medium or long.')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Lightning checkpoint path. Required.')
    parser.add_argument('--max_context', type=int, default=512,
                        help='Maximum context length.')
    parser.add_argument('--save_prefix', type=str, default=None,
                        help='Prefix of the save directory name. Default: amets. For example, "moe" saves to moe_<dataset>.')
    parser.add_argument('--max_samples', type=int, default=5000,
                        help='Maximum number of samples to extract.')
    parser.add_argument('--output_dir', type=str, default='router_embeddings',
                        help='Output directory for embeddings.')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on.')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Parse dataset(s). Comma-separated names select multi-dataset mode.
    dataset_names = [d.strip() for d in args.dataset.split(',')]
    is_multi = len(dataset_names) > 1

    if is_multi:
        print(f"\nMulti-dataset mode: {len(dataset_names)} datasets")
        for d in dataset_names:
            print(f"  - {d}")
        # Divide max_samples evenly across datasets
        samples_per_dataset = max(1, args.max_samples // len(dataset_names))
        print(f"Samples per dataset: {samples_per_dataset}")

    if args.checkpoint is None:
        raise ValueError("--checkpoint is required")

    model = load_amets_model(args.checkpoint, device)

    if is_multi:
        # Extract from each dataset and concatenate
        all_embeddings_list = []
        for ds_name in dataset_names:
            print(f"\n{'='*60}")
            print(f"Extracting from: {ds_name}")
            print(f"{'='*60}")
            emb = extract_amets_router_embeddings(
                model,
                dataset_name=ds_name,
                term=args.term,
                max_context=args.max_context,
                max_samples=samples_per_dataset,
                device=device,
            )
            all_embeddings_list.append(emb)

        # Concatenate embeddings across datasets
        embeddings = _concatenate_embeddings(all_embeddings_list)
        # Use combined name for saving
        save_dataset_name = "multi_" + "_".join(dataset_names[:3])
        if len(dataset_names) > 3:
            save_dataset_name += f"_+{len(dataset_names)-3}more"
    else:
        embeddings = extract_amets_router_embeddings(
            model,
            dataset_name=dataset_names[0],
            term=args.term,
            max_context=args.max_context,
            max_samples=args.max_samples,
            device=device,
        )
        save_dataset_name = dataset_names[0]

    # Save embeddings
    print(f"\nSaving embeddings to {args.output_dir}")
    save_prefix = args.save_prefix if args.save_prefix else 'amets'
    save_embeddings(embeddings, args.output_dir, save_prefix, save_dataset_name)

    print("\nExtraction complete.")


if __name__ == '__main__':
    main()
