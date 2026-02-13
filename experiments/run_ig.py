"""run integrated gradients experiments"""
import os
import argparse
import pandas as pd
import numpy as np
import torch

from src import (
    PROTEINS, load_mutations,
    load_esm_local, load_sae, tokenize_seq,
    integrated_gradients_mutation, topk_features
)

# auto-detect project directory
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_ig_for_mutation_layer(mutation_data, layer, esm_model, sae_model,
                                alphabet, batch_converter, device, ig_steps=10):
    """run IG for one mutation at one layer"""
    # tokenize sequences
    wt_seq = mutation_data['wt_seq']
    mut_seq = mutation_data['mutated_sequence']
    position = mutation_data['position']
    wt_aa = mutation_data['wt_residue']
    mut_aa = mutation_data['mut_residue']

    wt_tokens = tokenize_seq(wt_seq, batch_converter, device)
    mut_tokens = tokenize_seq(mut_seq, batch_converter, device)

    # run IG
    print(f"  running IG for {mutation_data['mutation']} at layer {layer}")
    effects_sae, info = integrated_gradients_mutation(
        esm_model=esm_model,
        sae_model=sae_model,
        wt_tokens=wt_tokens,
        mut_tokens=mut_tokens,
        position=position,
        wt_aa=wt_aa,
        mut_aa=mut_aa,
        alphabet=alphabet,
        hook_layer=layer,
        steps=ig_steps,
        device=device
    )

    # get top features
    top_abs = topk_features(effects_sae, k=20, mode='abs')
    top_pos = topk_features(effects_sae, k=10, mode='pos')
    top_neg = topk_features(effects_sae, k=10, mode='neg')

    # results
    result = {
        'protein': mutation_data['protein'],
        'mutation': mutation_data['mutation'],
        'position': position,
        'wt_aa': wt_aa,
        'mut_aa': mut_aa,
        'layer': layer,
        'dms_score': mutation_data['dms_score'],
        'esm_score': mutation_data['esm_score'],
        'baseline_score': info['baseline_score'],
        'top_20_abs_indices': top_abs['indices'],
        'top_20_abs_effects': top_abs['effects'],
        'top_10_pos_indices': top_pos['indices'],
        'top_10_pos_effects': top_pos['effects'],
        'top_10_neg_indices': top_neg['indices'],
        'top_10_neg_effects': top_neg['effects'],
        'total_effect': top_abs['total_effect'],
        'top_20_fraction': top_abs['top_k_fraction'],
        'all_effects': effects_sae.numpy(),  # full attribution vector (4096,)
    }

    return result


def save_result(result, output_dir):
    """
    save a single IG result progressively
    saves to: {output_dir}/{protein}_{mutation}_layer{layer}_ig.npz
    """
    protein = result['protein']
    mutation = result['mutation'].replace('>', '_')
    layer = result['layer']

    filename = f"{protein}_{mutation}_layer{layer}_ig.npz"
    filepath = os.path.join(output_dir, filename)

    # save as compressed numpy
    np.savez_compressed(filepath, **result)

    print(f"  saved to {filename}")


def main():
    parser = argparse.ArgumentParser(description='run integrated gradients experiments')
    parser.add_argument('--protein', required=True,
                       choices=['EPHB2_HUMAN', 'DNJA1_HUMAN', 'PR40A_HUMAN'])
    parser.add_argument('--n_mutations', type=int, default=100)
    parser.add_argument('--layers', type=str, default='24',
                       help='comma-separated layer indices, e.g., "24" or "8,12,16,20,24"')
    parser.add_argument('--ig_steps', type=int, default=10)
    parser.add_argument('--output_dir', default=None,
                       help='output directory (default: PROJECT_DIR/results/ig)')
    parser.add_argument('--esm_model_path', default='/datasets/bio/esm/models/esm2_t33_650M_UR50D.pt',
                       help='path to ESM-2 model checkpoint')
    parser.add_argument('--sae_weights_dir', default=None,
                       help='directory containing SAE weights (default: PROJECT_DIR/sae_weights)')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # set defaults based on PROJECT_DIR
    if args.output_dir is None:
        args.output_dir = f'{PROJECT_DIR}/results/ig'
    if args.sae_weights_dir is None:
        args.sae_weights_dir = f'{PROJECT_DIR}/sae_weights'

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device)
    layers = [int(x) for x in args.layers.split(',')]

    print(f"\n{'='*60}")
    print(f"integrated gradients experiment")
    print(f"{'='*60}")
    print(f"protein: {args.protein}")
    print(f"mutations: {args.n_mutations}")
    print(f"layers: {layers}")
    print(f"IG steps: {args.ig_steps}")
    print(f"device: {device}")
    print(f"output: {args.output_dir}")
    print(f"{'='*60}\n")

    # load mutation data
    df_mutations = load_mutations(args.protein, n_mutations=args.n_mutations, seed=args.seed)

    # load models (once)
    print(f"loading esm-2 from {args.esm_model_path}")
    esm_model, alphabet, batch_converter = load_esm_local(args.esm_model_path, device)

    protein_info = PROTEINS[args.protein]
    wt_seq = protein_info['wt_seq']

    # run IG for each mutation and layer
    for idx, row in df_mutations.iterrows():
        position = int(row['position'])
        wt_aa = row['wt_residue']
        mut_aa = row['mut_residue']
        mut_seq = row['mutated_sequence']
        dms_score = row['DMS_score']
        esm_score = row['esm2_t33_650M_UR50D']
        mutation_str = f"{wt_aa}{position}{mut_aa}"

        print(f"\n{mutation_str} (DMS={dms_score:.2f}, ESM={esm_score:.2f})")

        mutation_data = {
            'protein': args.protein,
            'mutation': mutation_str,
            'wt_seq': wt_seq,
            'mutated_sequence': mut_seq,
            'position': position,
            'wt_residue': wt_aa,
            'mut_residue': mut_aa,
            'dms_score': dms_score,
            'esm_score': esm_score
        }

        for layer in layers:
            # load sae for this layer
            sae_model = load_sae(layer, args.sae_weights_dir, device)

            # run IG
            result = run_ig_for_mutation_layer(
                mutation_data, layer, esm_model, sae_model,
                alphabet, batch_converter, device, args.ig_steps
            )

            # save progressively
            save_result(result, args.output_dir)

    print(f"\n{'-'*60}")
    print(f"results saved to {args.output_dir}")
    print(f"{'-'*60}")


if __name__ == '__main__':
    main()
