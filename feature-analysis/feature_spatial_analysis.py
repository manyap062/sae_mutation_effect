"""analyze spatial activation patterns of top IG features across WT sequences"""
import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import torch

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from src import (
    PROTEINS, load_esm_local, load_sae, tokenize_seq, get_logits_and_hidden
)


def load_ig_results(protein, layer, ig_dir):
    """load all IG .npz files for a protein/layer, return list of dicts"""
    pattern = os.path.join(ig_dir, f"{protein}_*_layer{layer}_ig.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no IG files found")
    print(f"found {len(files)} IG files for {protein} layer {layer}")
    results = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        results.append({k: d[k] for k in d.keys()})
    return results


def get_top_features_across_mutations(ig_results, k=20):
    """
    aggregate IG attributions across mutations, return top k features by mean absolute attribution
    """
    # stack all_effects: (n_mutations, 4096)
    all_effects = np.stack([r['all_effects'] for r in ig_results])
    mean_abs = np.mean(np.abs(all_effects), axis=0)  # (4096,)
    top_idx = np.argsort(mean_abs)[::-1][:k]
    top_attrs = mean_abs[top_idx]
    return top_idx, top_attrs


def get_wt_sae_activations(wt_seq, layer, esm_model, sae_model,
                           batch_converter, device):
    """
    run WT sequence through ESM -> SAE encode -> topK to get post-activation at every position

    returns: (seq_len, 4096) array of post-topk activations (seq_len = number of residues, excluding first/last token)
    """
    tokens = tokenize_seq(wt_seq, batch_converter, device)
    _, hidden = get_logits_and_hidden(esm_model, tokens, layer)
    # hidden: (1, L+2, 1280) where L = len(wt_seq), +2 for BOS/EOS
    seq_len = len(wt_seq)

    # encode all positions at once (positions 1..seq_len are residues)
    # BOS=0, residues=1..L, EOS=L+1
    residue_hidden = hidden[0, 1:seq_len+1, :]  # (seq_len, 1280)

    with torch.no_grad():
        pre_acts, mu, std = sae_model.encode(residue_hidden)
        # apply topk to get actual activations
        post_acts = sae_model.topK_activation(pre_acts, sae_model.k)

    return post_acts.cpu().numpy()  # (seq_len, 4096)


def count_contiguous_runs(positions):
    """count number of contiguous runs in a sorted list of positions"""
    if len(positions) == 0:
        return 0
    runs = 1
    for i in range(1, len(positions)):
        if positions[i] - positions[i-1] > 1:
            runs += 1
    return runs


def classify_feature(n_active, n_runs, seq_len):
    """classify feature into spatial category"""
    if n_active <= 2:
        return "single-site"
    elif n_active <= 8 and n_runs <= 2:
        return "local-window"
    else:
        return "distributed"


def analyze_protein_layer(protein, layer, ig_dir, esm_model, sae_model,
                          alphabet, batch_converter, device, top_k=20):
    """run full spatial analysis for one protein/layer"""
    wt_seq = PROTEINS[protein]['wt_seq']
    seq_len = len(wt_seq)

    # load IG results and get top features
    ig_results = load_ig_results(protein, layer, ig_dir)
    top_features, top_attrs = get_top_features_across_mutations(ig_results, k=top_k)

    # get WT activation pattern across all positions
    post_acts = get_wt_sae_activations(
        wt_seq, layer, esm_model, sae_model, batch_converter, device
    )  # (seq_len, 4096)

    # collect mutation positions from IG results
    mutation_positions = set()
    for r in ig_results:
        mutation_positions.add(int(r['position']))

    # analyze each top feature
    rows = []
    for rank, (feat_idx, ig_attr) in enumerate(zip(top_features, top_attrs)):
        feat_acts = post_acts[:, feat_idx]  # (seq_len,)
        active_mask = feat_acts > 0
        active_positions = np.where(active_mask)[0]
        # convert to 1-indexed (matching mutation position convention)
        active_positions_1idx = (active_positions + 1).tolist()
        # amino acids at active positions (0-indexed into wt_seq)
        active_aas = ''.join(wt_seq[i] for i in active_positions)
        n_active = len(active_positions)
        frac_active = n_active / seq_len
        n_runs = count_contiguous_runs(sorted(active_positions.tolist()))

        # check if feature fires at any mutation site
        active_at_mut = any(
            p in active_positions_1idx for p in mutation_positions
        )

        category = classify_feature(n_active, n_runs, seq_len)

        rows.append({
            'protein': protein,
            'layer': layer,
            'rank': rank + 1,
            'feature_idx': int(feat_idx),
            'ig_attribution': float(ig_attr),
            'n_positions_active': n_active,
            'frac_positions_active': round(frac_active, 4),
            'n_contiguous_runs': n_runs,
            'active_at_mutation_site': active_at_mut,
            'active_positions': str(active_positions_1idx),
            'active_aas': active_aas,
            'category': category,
        })

    df = pd.DataFrame(rows)
    df = add_flanking_to_df(df, PROTEINS)
    return df

def analyze_flanking_context(wt_seq, active_positions_1idx, target_aa, window=3):
    """
    compare flanking residues at positions where feature fires vs doesn't fire for a given amino acid
    """
    all_target_pos = [i for i, aa in enumerate(wt_seq) if aa == target_aa]
    active_set = set(p - 1 for p in active_positions_1idx)  # convert to 0-index

    active_contexts = []
    inactive_contexts = []

    for pos in all_target_pos:
        left = max(0, pos - window)
        right = min(len(wt_seq), pos + window + 1)
        ctx = '-' * (window - (pos - left)) + wt_seq[left:right] + '-' * (window - (right - pos - 1))

        entry = (pos + 1, ctx)  # 1-index
        if pos in active_set:
            active_contexts.append(entry)
        else:
            inactive_contexts.append(entry)

    return active_contexts, inactive_contexts, len(all_target_pos)


def add_flanking_to_df(df, proteins_dict, window=3):
    """add flanking context columns to the spatial analysis dataframe"""
    flanking_rows = []
    for _, row in df.iterrows():
        wt_seq = proteins_dict[row['protein']]['wt_seq']
        active_positions = eval(row['active_positions'])  # list of 1-indexed
        target_aa = row['active_aas']

        # only add if feature fires on a single AA
        unique_aas = set(target_aa)
        if len(unique_aas) != 1 or len(target_aa) == 0:
            flanking_rows.append({
                'single_aa': False,
                'target_aa': None,
                'n_total_instances': None,
                'n_active': None,
                'n_inactive': None,
                'selectivity': None,
                'active_contexts': None,
                'inactive_contexts': None,
            })
            continue

        aa = unique_aas.pop()
        active_ctx, inactive_ctx, n_total = analyze_flanking_context(
            wt_seq, active_positions, aa, window
        )

        flanking_rows.append({
            'single_aa': True,
            'target_aa': aa,
            'n_total_instances': n_total,
            'n_active': len(active_ctx),
            'n_inactive': len(inactive_ctx),
            'selectivity': f"{len(active_ctx)}/{n_total}",
            'active_contexts': str([c[1] for c in active_ctx]),
            'inactive_contexts': str([c[1] for c in inactive_ctx]),
        })

    flanking_df = pd.DataFrame(flanking_rows)
    return pd.concat([df.reset_index(drop=True), flanking_df], axis=1)


def print_summary(df):
    """category breakdown per protein"""
    for protein in df['protein'].unique():
        sub = df[df['protein'] == protein]
        counts = sub['category'].value_counts()
        print(f"\n{protein} (layer {sub['layer'].iloc[0]}):")
        for cat in ['single-site', 'local-window', 'distributed']:
            n = counts.get(cat, 0)
            print(f"  {cat:15s}: {n:3d} / {len(sub)} features")

    # overall
    print(f"\noverall:")
    counts = df['category'].value_counts()
    for cat in ['single-site', 'local-window', 'distributed']:
        n = counts.get(cat, 0)
        print(f"  {cat:15s}: {n:3d} / {len(df)} features")

    # summary stats
    print(f"\nmean positions active: {df['n_positions_active'].mean():.1f}")
    print(f"mean frac active:     {df['frac_positions_active'].mean():.3f}")
    print(f"features active at mutation site: "
          f"{df['active_at_mutation_site'].sum()} / {len(df)}")
    
    # flanking context summary
    single_aa = df[df['single_aa'] == True]
    if len(single_aa) > 0:
        perfect = single_aa[single_aa['n_active'] == single_aa['n_total_instances']]
        selective = single_aa[single_aa['n_active'] < single_aa['n_total_instances']]
        print(f"\nsingle-AA features: {len(single_aa)} / {len(df)}")
        print(f"  pure AA detectors (fire at ALL instances): {len(perfect)}")
        print(f"  context-selective (skip some instances):   {len(selective)}")
        if len(selective) > 0:
            print(f"\n  context-selective features:")
            for _, row in selective.iterrows():
                print(f"    feature {row['feature_idx']} ({row['target_aa']}): "
                      f"{row['selectivity']} instances")
                print(f"      active:   {row['active_contexts']}")
                print(f"      inactive: {row['inactive_contexts']}")


def main():
    parser = argparse.ArgumentParser(
        description='analyze spatial activation patterns of top IG features'
    )
    parser.add_argument('--protein', default=None,
                        choices=['EPHB2_HUMAN', 'DNJA1_HUMAN', 'PR40A_HUMAN'],
                        help='protein to analyze (default: all three)')
    parser.add_argument('--layer', type=int, default=16)
    parser.add_argument('--top_k', type=int, default=20)
    parser.add_argument('--ig_dir', default=None,
                        help='IG results directory (default: PROJECT_DIR/results/ig)')
    parser.add_argument('--output_dir', default=None,
                        help='output directory (default: PROJECT_DIR/results/spatial_analysis)')
    parser.add_argument('--esm_model_path',
                        default='/datasets/bio/esm/models/esm2_t33_650M_UR50D.pt')
    parser.add_argument('--sae_weights_dir', default=None,
                        help='SAE weights directory (default: PROJECT_DIR/sae_weights)')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    if args.ig_dir is None:
        args.ig_dir = os.path.join(PROJECT_DIR, 'results', 'ig')
    if args.output_dir is None:
        args.output_dir = os.path.join(PROJECT_DIR, 'results', 'spatial_analysis')
    if args.sae_weights_dir is None:
        args.sae_weights_dir = os.path.join(PROJECT_DIR, 'sae_weights')

    os.makedirs(args.output_dir, exist_ok=True)

    proteins = [args.protein] if args.protein else ['EPHB2_HUMAN', 'DNJA1_HUMAN', 'PR40A_HUMAN']

    print(f"loading ESM-2 from {args.esm_model_path}")
    esm_model, alphabet, batch_converter = load_esm_local(args.esm_model_path, args.device)

    all_dfs = []
    for protein in proteins:
        print(f"analyzing {protein} layer {args.layer}")
        print(f"{'-'*60}")

        sae_model = load_sae(args.layer, args.sae_weights_dir, args.device)

        df = analyze_protein_layer(
            protein, args.layer, args.ig_dir,
            esm_model, sae_model, alphabet, batch_converter,
            args.device, top_k=args.top_k
        )
        all_dfs.append(df)

        # save per-protein csv
        out_path = os.path.join(
            args.output_dir, f"{protein}_layer{args.layer}_spatial.csv"
        )
        df.to_csv(out_path, index=False)
        print(f"saved {out_path}")

    combined = pd.concat(all_dfs, ignore_index=True)
    combined_path = os.path.join(
        args.output_dir, f"all_proteins_layer{args.layer}_spatial.csv"
    )
    combined.to_csv(combined_path, index=False)
    print(f"\ncombined results saved to {combined_path}")

    print_summary(combined)


if __name__ == '__main__':
    main()
