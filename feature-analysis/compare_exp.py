import pandas as pd
import numpy as np 
4-32
8-24
ig_dfs = []
for layer in range(8, 25, 4):
    df = pd.read_csv(f"/project/pi_annagreen_umass_edu/manya/sae_mutation_effect/ig_code/ig_results/ig_feature_consistency_layer{layer}.csv")
    df['layer'] = layer
    ig_dfs.append(df)

ig_all = pd.concat(ig_dfs, ignore_index=True)

#top IG = features that appear in top20 most often
def get_top_ig_by_frequency(ig_df, layer, top_n=20):
    layer_data = ig_df[ig_df['layer'] == layer]
    return set(layer_data.nlargest(top_n, 'frequency')['feature_idx'])

def get_top_ig_by_total(ig_df, layer, top_n=20):
    layer_data = ig_df[ig_df['layer'] == layer].copy()
    layer_data['abs_mean_effect'] = layer_data['mean_effect'].abs()
    layer_data['total_attribution'] = layer_data['frequency'] * layer_data['abs_mean_effect']
    return set(layer_data.nlargest(top_n, 'total_attribution')['feature_idx'])
    
patching_dfs = []
for protein in ['EPHB2_HUMAN', 'DNJA1_HUMAN', 'PR40A_HUMAN']:
    for layer in range(8, 25, 4):
        df = pd.read_csv(f"/project/pi_annagreen_umass_edu/manya/sae_mutation_effect/patching_results_4/{protein}_layer{layer}_seed42_patching1.csv")
        patching_dfs.append(df)

patching_all = pd.concat(patching_dfs, ignore_index=True)
patching_real = patching_all[patching_all['control_type'] == 'real']

# top patching = features that have largest average score change
def get_top_patching_by_effect(patching_df, protein, layer, top_n=20):
    subset = patching_df[(patching_df['protein'] == protein) & 
                         (patching_df['layer'] == layer)]
    feature_effects = subset.groupby('feature_idx')['score_change'].apply(
        lambda x: np.abs(x).mean()
    ).reset_index()
    feature_effects.columns = ['feature_idx', 'mean_abs_effect']
    return set(feature_effects.nlargest(top_n, 'mean_abs_effect')['feature_idx'])

def compare_ig_vs_patching(ig_df, patching_df, top_n=20):
    """
    Compare IG and patching features.
    """

    results = []
    
    for layer in range(8, 25, 4):
        ig_features = get_top_ig_by_total(ig_df, layer, top_n)
        
        if len(ig_features) == 0:
            continue
        
        for protein in ['EPHB2_HUMAN', 'DNJA1_HUMAN', 'PR40A_HUMAN']:
            patching_features = get_top_patching_by_effect(patching_df, protein, layer, top_n)
            
            if len(patching_features) == 0:
                continue
            
            overlap = ig_features & patching_features
            ig_only = ig_features - patching_features
            patching_only = patching_features - ig_features
            
            results.append({
                'layer': layer,
                'protein': protein,
                'n_ig': len(ig_features),
                'n_patching': len(patching_features),
                'n_overlap': len(overlap),
                'overlap_pct': len(overlap) / top_n * 100,
                'jaccard': len(overlap) / len(ig_features | patching_features),
                'ig_only': len(ig_only),
                'patching_only': len(patching_only)
            })
    
    return pd.DataFrame(results)

results = compare_ig_vs_patching(ig_all, patching_real, top_n=20)
print(f"\nMean overlap: {results['overlap_pct'].mean():.1f}%")
print(f"Median overlap: {results['overlap_pct'].median():.1f}%")
print(f"Range: [{results['overlap_pct'].min():.1f}%, {results['overlap_pct'].max():.1f}%]")

# layer 16 analysis
layer16 = results[results['layer'] == 16]
print(f"\nLayer 16:")
print(f"  Mean overlap: {layer16['overlap_pct'].mean():.1f}%")
print(layer16[['protein', 'n_overlap', 'overlap_pct']].to_string(index=False))  

# IG top20 for DNJA1 layer 16
ig_layer16 = ig_all[ig_all['layer'] == 16]
ig_top20 = set(ig_layer16.nlargest(20, 'frequency')['feature_idx'])

# Get patching features for DNJA1 layer 16
dnja1_layer16 = patching_real[(patching_real['protein'] == 'DNJA1_HUMAN') & 
                               (patching_real['layer'] == 16)]

all_patched_features = set(dnja1_layer16['feature_idx'].unique())
ig_features_that_were_patched = ig_top20 & all_patched_features

ig_layer16['abs_mean_effect'] = ig_layer16['mean_effect'].abs()
ig_layer16['total_attribution'] = ig_layer16['frequency'] * ig_layer16['abs_mean_effect']

print(f"IG top-20 features: {len(ig_top20)}")
print(f"IG features that were patched: {len(ig_features_that_were_patched)}")
print(f"IG features not patched: {len(ig_top20 - all_patched_features)}")

print("T20 by FREQUENCY:")
print(ig_layer16.nlargest(20, 'frequency')[['feature_idx', 'frequency', 'mean_effect', 'count']])

print("\nT20 by MEAN EFFECT:")
print(ig_layer16.nlargest(20, 'abs_mean_effect')[['feature_idx', 'frequency', 'mean_effect', 'abs_mean_effect']])

print("\nT20 by TOTAL (freq* effect):")
print(ig_layer16.nlargest(20, 'total_attribution')[['feature_idx', 'frequency', 'abs_mean_effect', 'total_attribution']])
