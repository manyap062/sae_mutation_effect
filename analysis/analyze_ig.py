"""analyze integrated gradients results"""
import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict


def load_all_ig_results(results_dir):
    """
    load all IG .npz files from a directory
    returns list of result dictionaries
    """
    results = []

    for filename in os.listdir(results_dir):
        if not filename.endswith('_ig.npz'):
            continue

        filepath = os.path.join(results_dir, filename)
        data = np.load(filepath, allow_pickle=True)

        result = {k: data[k] for k in data.files}
        results.append(result)

    print(f"loaded {len(results)} IG results from {results_dir}")
    return results


def compute_feature_consistency(results, layer, top_k=20):
    """
    find features that appear in top-k most frequently across mutations at a given layer

    args:
        results: list of IG result dicts
        layer: which layer to analyze
        top_k: consider top-k features per mutation

    returns:
        dataframe with feature consistency statistics
    """
    # filter to specific layer
    layer_results = [r for r in results if r['layer'].item() == layer]

    if not layer_results:
        print(f"no results found for layer {layer}")
        return pd.DataFrame()

    # count how many times each feature appears in top-k
    feature_counts = defaultdict(int)
    feature_effects = defaultdict(list)

    for result in layer_results:
        top_indices = result['top_20_abs_indices'][:top_k]
        top_effects = result['top_20_abs_effects'][:top_k]

        for idx, effect in zip(top_indices, top_effects):
            feature_counts[int(idx)] += 1
            feature_effects[int(idx)].append(float(effect))

    # convert to dataframe
    consistency_data = []
    for feat_idx, count in feature_counts.items():
        consistency_data.append({
            'feature_idx': feat_idx,
            'count': count,
            'frequency': count / len(layer_results),
            'mean_effect': np.mean(feature_effects[feat_idx]),
            'std_effect': np.std(feature_effects[feat_idx]),
            'median_effect': np.median(feature_effects[feat_idx])
        })

    df = pd.DataFrame(consistency_data)
    df = df.sort_values('count', ascending=False)

    return df


def compute_layer_statistics(results):
    """
    aggregate statistics per layer: sparsity, effect magnitudes, etc.
    returns dataframe with one row per layer
    """
    layer_stats = []

    for layer in sorted(set(r['layer'].item() for r in results)):
        layer_results = [r for r in results if r['layer'].item() == layer]

        # aggregate metrics
        top_20_fractions = [r['top_20_fraction'].item() for r in layer_results]
        total_effects = [r['total_effect'].item() for r in layer_results]
        baseline_scores = [r['baseline_score'].item() for r in layer_results]

        # compute sparsity: what fraction of 4096 features are near-zero?
        all_effects = np.array([r['all_effects'] for r in layer_results])  # (n_mutations, 4096)
        near_zero = (np.abs(all_effects) < 0.01).sum(axis=1).mean()  # features with |effect| < 0.01
        sparsity = near_zero / 4096

        layer_stats.append({
            'layer': layer,
            'n_mutations': len(layer_results),
            'mean_top20_fraction': np.mean(top_20_fractions),
            'std_top20_fraction': np.std(top_20_fractions),
            'mean_total_effect': np.mean(total_effects),
            'mean_baseline_score': np.mean(baseline_scores),
            'sparsity': sparsity,
        })

    return pd.DataFrame(layer_stats)


def find_top_features_per_layer(results, top_k=10):
    """
    for each layer, find the most consistently important features

    returns:
        dict: {layer: {'feature_idx': [list], 'frequency': [list]}}
    """
    layers = sorted(set(r['layer'].item() for r in results))
    top_features = {}

    for layer in layers:
        consistency_df = compute_feature_consistency(results, layer, top_k=20)

        if consistency_df.empty:
            continue

        top_k_features = consistency_df.head(top_k)

        top_features[layer] = {
            'feature_idx': top_k_features['feature_idx'].tolist(),
            'frequency': top_k_features['frequency'].tolist(),
            'mean_effect': top_k_features['mean_effect'].tolist()
        }

    return top_features


def protein_comparison(results, layer):
    """
    compare feature usage across proteins at a given layer
    returns dataframe showing which features are protein-specific vs shared
    """
    proteins = sorted(set(str(r['protein'].item()) for r in results))

    protein_features = {}
    for protein in proteins:
        protein_layer_results = [
            r for r in results
            if r['layer'].item() == layer and str(r['protein'].item()) == protein
        ]

        if not protein_layer_results:
            continue

        # get top-20 features across all mutations for this protein
        feature_counts = defaultdict(int)
        for r in protein_layer_results:
            for idx in r['top_20_abs_indices'][:20]:
                feature_counts[int(idx)] += 1

        # keep features that appear in >50% of mutations
        threshold = len(protein_layer_results) * 0.5
        protein_features[protein] = set(
            idx for idx, count in feature_counts.items() if count >= threshold
        )

    # find shared vs unique features
    comparison_data = []

    if len(protein_features) >= 2:
        all_features = set.union(*protein_features.values())

        for feat_idx in all_features:
            appears_in = [p for p, feats in protein_features.items() if feat_idx in feats]
            comparison_data.append({
                'feature_idx': feat_idx,
                'proteins': ', '.join(appears_in),
                'n_proteins': len(appears_in),
                'is_shared': len(appears_in) > 1
            })

    df = pd.DataFrame(comparison_data)
    if not df.empty:
        df = df.sort_values('n_proteins', ascending=False)

    return df


def main(results_dir):
    """main analysis pipeline"""

    # load all results
    results = load_all_ig_results(results_dir)

    if not results:
        print("no results found!")
        return

    print("\n" + "-"*50)
    print("INTEGRATED GRADIENTS ANALYSIS")
    print("-"*50)

    # 1. layer-wise statistics
    print("\n1. layer-wise statistics")
    print("-" * 60)
    layer_stats = compute_layer_statistics(results)
    print(layer_stats.to_string(index=False))

    # 2. top features per layer
    print("\n2. most consistent features per layer")
    print("-" * 60)
    top_features = find_top_features_per_layer(results, top_k=10)
    for layer, data in top_features.items():
        print(f"\nlayer {layer}:")
        for idx, freq, effect in zip(data['feature_idx'][:5],
                                     data['frequency'][:5],
                                     data['mean_effect'][:5]):
            print(f"  feature {idx:4d}: {freq*100:5.1f}% frequency, "
                  f"mean effect = {effect:+.3f}")

    # 3. feature consistency (detailed for one layer)
    available_layers = [r['layer'].item() for r in results]
    if 16 in available_layers:
        test_layer = 16
    elif 24 in available_layers:
        test_layer = 24
    else:
        test_layer = available_layers[0]

    print(f"\n3. feature consistency at layer {test_layer} (top 20)")
    print("-" * 60)
    consistency = compute_feature_consistency(results, layer=test_layer, top_k=20)
    print(consistency.head(20).to_string(index=False))

    # 4. protein comparison (if multiple proteins)
    proteins = set(str(r['protein'].item()) for r in results)
    if len(proteins) > 1 and test_layer in available_layers:
        print(f"\n4. protein comparison at layer {test_layer}")
        print("-" * 60)
        comparison = protein_comparison(results, layer=test_layer)
        if not comparison.empty:
            print(f"\nshared features (appear in >1 protein):")
            shared = comparison[comparison['is_shared']]
            print(shared.head(10).to_string(index=False))

    # 5. save summary csvs
    output_dir = results_dir
    layer_stats.to_csv(os.path.join(output_dir, 'ig_layer_statistics.csv'), index=False)
    print(f"\nsaved layer statistics to: ig_layer_statistics.csv")

    for layer in sorted(set(r['layer'].item() for r in results)):
        consistency = compute_feature_consistency(results, layer, top_k=100)
        consistency.to_csv(
            os.path.join(output_dir, f'ig_feature_consistency_layer{layer}.csv'),
            index=False
        )

    print("\n" + "-"*60)


if __name__ == '__main__':
    import os
    PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = (
        sys.argv[1]
        if len(sys.argv) > 1
        else f"{PROJECT_DIR}/results/ig"
    )
    main(results_dir)
