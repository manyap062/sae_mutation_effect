"""activation patching logic for single-feature interventions"""
import torch
import numpy as np
from .scoring import wt_marginal_score


def forward_from_layer(model, hidden_states, start_layer):
    """
    forward pass from a specific layer onwards using esm internals
    continue through remaining transformer layers
    """
    x = hidden_states

    # continue through remaining transformer layers
    for layer_idx in range(start_layer, model.num_layers):
        x, _ = model.layers[layer_idx](x, self_attn_padding_mask=None)

    # final layer norm
    x = model.emb_layer_norm_after(x)

    # lm head
    logits = model.lm_head(x)

    return logits


def get_sae_activations(sae, hidden_states, position):
    """get sae pre-activations at specific position"""
    h = hidden_states[0, position]  # (1280,)
    pre_acts, mu, std = sae.encode(h.unsqueeze(0))
    return pre_acts.squeeze(0), mu, std


def run_patching_experiment(model, alphabet, batch_converter, sae, layer_idx,
                            wt_seq, mut_seq, position, wt_aa, mut_aa, device,
                            tokenize_fn, get_logits_hidden_fn, top_k=5, control_type='real'):
    """
    run activation patching for one mutation at one layer

    both baseline and patched MUST use same sae encode/decode path
    or you get ~9 logodds when it should be 0 for no-op control

    args:
        model: ESM-2 model
        alphabet: ESM alphabet
        batch_converter: ESM batch converter
        sae: SparseAutoencoder instance
        layer_idx: layer to patch at
        wt_seq: wild-type sequence
        mut_seq: mutant sequence
        position: 1-indexed mutation position
        wt_aa: wild-type amino acid
        mut_aa: mutant amino acid
        device: torch device
        tokenize_fn: function to tokenize sequences
        get_logits_hidden_fn: function to get logits and hidden states
        top_k: number of top features to patch (default 5)
        control_type: 'real' (wt→mut), 'noop' (wt→wt), 'random' (stable features)

    returns:
        dict with baseline_score, feature_deltas, patched_results
    """
    # tokenize
    wt_tokens = tokenize_fn(wt_seq, batch_converter, device)
    mut_tokens = tokenize_fn(mut_seq, batch_converter, device)

    tok_pos = position  # 1-indexed = token position (BOS at 0)

    # get hidden states at target layer
    wt_logits, wt_hidden = get_logits_hidden_fn(model, wt_tokens, layer_idx)
    _, mut_hidden = get_logits_hidden_fn(model, mut_tokens, layer_idx)

    # get sae activations at mutation position
    wt_pre_acts, wt_mu, wt_std = get_sae_activations(sae, wt_hidden, tok_pos)
    mut_pre_acts, _, _ = get_sae_activations(sae, mut_hidden, tok_pos)

    # compute feature deltas
    deltas = (mut_pre_acts - wt_pre_acts).cpu().numpy()

    # baseline: reconstruct wt through sae (critical for fair comparison)
    wt_hidden_reconstructed = wt_hidden.clone()
    wt_hidden_reconstructed[0, tok_pos] = sae.decode(
        wt_pre_acts.unsqueeze(0), wt_mu, wt_std
    ).squeeze(0)
    baseline_logits = forward_from_layer(model, wt_hidden_reconstructed, layer_idx)
    baseline_score = wt_marginal_score(baseline_logits, position, wt_aa, mut_aa, alphabet)

    # select features to patch based on control type
    if control_type == 'noop':
        # no-op control: patch wt → wt (should yield ~0 change)
        top_indices = np.argsort(np.abs(deltas))[-top_k:][::-1]
        patch_values_list = [(idx, wt_pre_acts[idx]) for idx in top_indices]
    elif control_type == 'random':
        # random control: patch features that don't change much (bottom-k by |delta|)
        bottom_indices = np.argsort(np.abs(deltas))[:top_k]
        patch_values_list = [(idx, mut_pre_acts[idx]) for idx in bottom_indices]
    else:  # 'real'
        # real experiment: patch top-k changing features to mutant values
        top_indices = np.argsort(np.abs(deltas))[-top_k:][::-1]
        patch_values_list = [(idx, mut_pre_acts[idx]) for idx in top_indices]

    # patch each selected feature
    patched_scores = {}
    for feat_idx, patch_value in patch_values_list:
        # decode with patched feature
        patched_h = sae.decode_with_patch(
            wt_pre_acts.unsqueeze(0), wt_mu, wt_std,
            patch_indices=torch.tensor([feat_idx], device=device),
            patch_values=patch_value.reshape(1, 1)
        ).squeeze(0)

        # replace hidden state at mutation position
        wt_hidden_patched = wt_hidden.clone()
        wt_hidden_patched[0, tok_pos] = patched_h

        # forward through remaining layers
        patched_logits = forward_from_layer(model, wt_hidden_patched, layer_idx)

        # compute patched score
        patched_score = wt_marginal_score(patched_logits, position, wt_aa, mut_aa, alphabet)

        patched_scores[int(feat_idx)] = {
            'patched_score': patched_score,
            'score_change': patched_score - baseline_score,
            'delta': float(deltas[feat_idx]),
            'wt_act': float(wt_pre_acts[feat_idx].cpu()),
            'mut_act': float(mut_pre_acts[feat_idx].cpu()),
            'patch_value': float(patch_value.cpu()),
            'control_type': control_type
        }

    return {
        'baseline_score': baseline_score,
        'feature_deltas': deltas,
        'patched_results': patched_scores
    }
