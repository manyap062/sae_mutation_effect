"""sae mutation effect prediction library"""
from .sae import SparseAutoencoder
from .scoring import wt_marginal_score
from .patching import run_patching_experiment, forward_from_layer, get_sae_activations
from .ig import integrated_gradients_mutation, topk_features
from .data_utils import PROTEINS, load_mutations, parse_mutation_string
from .esm_utils import load_esm_local, load_sae, get_logits_and_hidden, tokenize_seq

__all__ = [
    'SparseAutoencoder',
    'wt_marginal_score',
    'run_patching_experiment',
    'forward_from_layer',
    'get_sae_activations',
    'integrated_gradients_mutation',
    'topk_features',
    'PROTEINS',
    'load_mutations',
    'parse_mutation_string',
    'load_esm_local',
    'load_sae',
    'get_logits_and_hidden',
    'tokenize_seq',
]
