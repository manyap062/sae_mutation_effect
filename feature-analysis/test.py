import pandas as pd
import numpy as np

df = pd.read_csv("/project/pi_annagreen_umass_edu/manya/sae_mutation_effect/patching_results_4/EPHB2_HUMAN_layer16_seed42_patching1.csv")
df_real = df[df['control_type'] == 'real']

# get mutation labels for each feature
feature_consistency = df_real.groupby('feature_idx').agg({
    'mutation': lambda x: ', '.join(x.unique()), 
    'score_change': lambda x: np.abs(x).mean()
}).reset_index()

# count mutations
feature_consistency['n_mutations'] = feature_consistency['mutation'].apply(lambda x: len(x.split(', ')))

total_mutations = df_real['mutation'].nunique()
feature_consistency['mutation_fraction'] = feature_consistency['n_mutations'] / total_mutations

# sort by effect size
feature_consistency = feature_consistency.sort_values('score_change', ascending=False)

print("Top 20 features by causal effect (EPHB2 Layer 16):")
print(feature_consistency[['feature_idx', 'mutation', 'score_change', 'mutation_fraction']].head(20))