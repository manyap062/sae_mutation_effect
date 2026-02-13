"""run activation patching experiments"""
import os
import argparse
import pandas as pd
import numpy as np
import torch

from src import (
    PROTEINS, load_mutations, parse_mutation_string,
    load_esm_local, load_sae, tokenize_seq, get_logits_and_hidden,
    run_patching_experiment
)

# auto-detect project directory
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protein', required=True, choices=['EPHB2_HUMAN', 'DNJA1_HUMAN', 'PR40A_HUMAN'])
    parser.add_argument('--layer', type=int, required=True)
    parser.add_argument('--output_dir', default=None,
                       help='output directory (default: PROJECT_DIR/results/patching)')
    parser.add_argument('--esm_model_path', default='/datasets/bio/esm/models/esm2_t33_650M_UR50D.pt',
                       help='path to ESM-2 model checkpoint')
    parser.add_argument('--sae_weights_dir', default=None,
                       help='directory containing SAE weights (default: PROJECT_DIR/sae_weights)')
    parser.add_argument('--n_mutations', type=int, default=100)
    parser.add_argument('--control_types', nargs='+', default=['real'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    # set defaults based on PROJECT_DIR
    if args.output_dir is None:
        args.output_dir = f'{PROJECT_DIR}/results/patching'
    if args.sae_weights_dir is None:
        args.sae_weights_dir = f'{PROJECT_DIR}/sae_weights'

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device)
    protein_info = PROTEINS[args.protein]

    # load mutation data
    df_mutations = load_mutations(args.protein, n_mutations=args.n_mutations, seed=args.seed)

    # load models
    print(f"Loading esm-2 from {args.esm_model_path}")
    model, alphabet, batch_converter = load_esm_local(args.esm_model_path, device)

    print(f"Loading sae for layer {args.layer}")
    sae = load_sae(args.layer, args.sae_weights_dir, device)

    wt_seq = protein_info['wt_seq']

    # run experiments
    results = []
    print(f"\nrunning patching on {len(df_mutations)} mutations for {args.protein} at layer {args.layer}")
    print(f"control types: {args.control_types}")

    for idx, row in df_mutations.iterrows():
        position = int(row['position'])
        wt_aa = row['wt_residue']
        mut_aa = row['mut_residue']
        mut_seq = row['mutated_sequence']
        dms_score = row['DMS_score']
        esm_score = row['esm2_t33_650M_UR50D']
        mutation_str = f"{wt_aa}{position}{mut_aa}"

        print(f"  {mutation_str} (DMS={dms_score:.2f}, ESM={esm_score:.2f})")

        # run patching for each control type
        for control_type in args.control_types:
            result = run_patching_experiment(
                model, alphabet, batch_converter, sae, args.layer,
                wt_seq, mut_seq, position, wt_aa, mut_aa, device,
                tokenize_fn=tokenize_seq,
                get_logits_hidden_fn=get_logits_and_hidden,
                top_k=5,
                control_type=control_type
            )

            # save results for each patched feature
            for feat_idx, feat_data in result['patched_results'].items():
                results.append({
                    'protein': args.protein,
                    'layer': args.layer,
                    'mutation': mutation_str,
                    'position': position,
                    'feature_idx': feat_idx,
                    'control_type': feat_data['control_type'],
                    'wt_pre_act': feat_data['wt_act'],
                    'mut_pre_act': feat_data['mut_act'],
                    'feature_delta': feat_data['delta'],
                    'patch_value': feat_data['patch_value'],
                    'baseline_score': result['baseline_score'],
                    'patched_score': feat_data['patched_score'],
                    'score_change': feat_data['score_change'],
                    'abs_score_change': abs(feat_data['score_change']),
                    'abs_feature_delta': abs(feat_data['delta']),
                    'dms_score': dms_score,
                    'esm_score': esm_score
                })

    # save results
    results_df = pd.DataFrame(results)
    output_filename = f"{args.protein}_layer{args.layer}_seed{args.seed}_patching1.csv"
    output_path = os.path.join(args.output_dir, output_filename)
    results_df.to_csv(output_path, index=False)
    print(f"\nsaved results to {output_path}")
    print(f"total rows: {len(results_df)}")


if __name__ == '__main__':
    main()
