"""integrated gradients for mutation effect prediction"""
import torch
import numpy as np
from typing import Tuple


def wt_marginal_score(logits: torch.Tensor, position: int, wt_aa: str,
                     mut_aa: str, alphabet) -> torch.Tensor:
    """
    compute wt marginal score: log p(mut_aa | x^wt) - log p(wt_aa | x^wt)

    args:
        logits: (1, L+2, vocab_size) model output
        position: 1-indexed mutation position (token index after BOS)
        wt_aa: wild-type amino acid (single letter)
        mut_aa: mutant amino acid (single letter)
        alphabet: ESM alphabet for AA→index conversion

    returns:
        scalar tensor: mutation effect score (negative = deleterious)
    """
    tok_pos = position  # 1-indexed position = token index (BOS at 0)
    log_probs = torch.log_softmax(logits[0, tok_pos], dim=-1)

    wt_idx = alphabet.get_idx(wt_aa)
    mut_idx = alphabet.get_idx(mut_aa)

    score = log_probs[mut_idx] - log_probs[wt_idx]
    return score


def forward_from_layer(model, hidden_states: torch.Tensor, start_layer: int) -> torch.Tensor:
    """
    forward pass from a specific layer through end of model

    args:
        model: ESM2 model
        hidden_states: (1, L+2, 1280) activations at start_layer
        start_layer: layer index to start from (0-32 for ESM-2-650M)

    returns:
        logits: (1, L+2, vocab_size)
    """
    x = hidden_states

    # continue through remaining transformer layers
    for layer_idx in range(start_layer, len(model.layers)):
        x, _ = model.layers[layer_idx](x, self_attn_padding_mask=None)

    x = model.emb_layer_norm_after(x)
    logits = model.lm_head(x)

    return logits


def integrated_gradients_mutation(
    esm_model,
    sae_model,
    wt_tokens: torch.Tensor,
    mut_tokens: torch.Tensor,
    position: int,
    wt_aa: str,
    mut_aa: str,
    alphabet,
    hook_layer: int,
    steps: int = 10,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, dict]:
    """
    run integrated gradients for one mutation at one layer

    args:
        esm_model: ESM-2 model
        sae_model: sparse autoencoder for target layer
        wt_tokens: (1, L+2) tokenized wild-type sequence
        mut_tokens: (1, L+2) tokenized mutant sequence
        position: 1-indexed mutation position
        wt_aa: wild-type amino acid
        mut_aa: mutant amino acid
        alphabet: ESM alphabet
        hook_layer: which layer to intervene at (0-32)
        steps: number of integration steps (default 10)
        device: 'cpu' or 'cuda'

    returns:
        effects_sae: (4096,) tensor of feature attributions at mutation position
        info: dict with baseline scores and diagnostics
    """

    # get hidden states at target layer for both WT and mutant
    with torch.no_grad():
        wt_results = esm_model(wt_tokens, repr_layers=[hook_layer], return_contacts=False)
        mut_results = esm_model(mut_tokens, repr_layers=[hook_layer], return_contacts=False)

        wt_hidden = wt_results["representations"][hook_layer]  # (1, L+2, 1280)
        mut_hidden = mut_results["representations"][hook_layer]

    # get sae pre-activations at mutation position
    tok_pos = position
    wt_h = wt_hidden[0, tok_pos]  # (1280,)
    mut_h = mut_hidden[0, tok_pos]

    with torch.no_grad():
        wt_pre_acts, wt_mu, wt_std = sae_model.encode(wt_h.unsqueeze(0))
        mut_pre_acts, _, _ = sae_model.encode(mut_h.unsqueeze(0))

        wt_pre_acts = wt_pre_acts.squeeze(0)  # (4096,)
        mut_pre_acts = mut_pre_acts.squeeze(0)

    # compute baseline wt score (for reporting)
    with torch.no_grad():
        # reconstruct wt hidden state through sae (same as α=0.00 in IG loop)
        wt_h_reconstructed = sae_model.decode(
            wt_pre_acts.unsqueeze(0),
            wt_mu,
            wt_std
        ).squeeze(0)

        # patch this into hidden states
        wt_hidden_baseline = wt_hidden.clone()
        wt_hidden_baseline[0, tok_pos] = wt_h_reconstructed

        # forward and compute score
        baseline_logits = forward_from_layer(esm_model, wt_hidden_baseline, hook_layer)
        baseline_score = wt_marginal_score(baseline_logits, position, wt_aa, mut_aa, alphabet)

    # run integrated gradients
    effects = []

    for step_idx, alpha in enumerate(np.linspace(0, 1, steps, endpoint=False)):
        # interpolate between wt and mutant pre-activations
        interpolated = (1 - alpha) * wt_pre_acts + alpha * mut_pre_acts
        interpolated = interpolated.to(device)
        interpolated.requires_grad_(True)

        # decode through sae
        interpolated_h = sae_model.decode_with_grad(
            interpolated.unsqueeze(0),
            wt_mu.to(device),
            wt_std.to(device)
        ).squeeze(0)

        # patch this into the hidden states at mutation position
        wt_hidden_patched = wt_hidden.clone().to(device)
        wt_hidden_patched[0, tok_pos] = interpolated_h

        # forward through remaining layers
        logits = forward_from_layer(esm_model, wt_hidden_patched, hook_layer)

        # compute score (this is our metric function)
        score = wt_marginal_score(logits, position, wt_aa, mut_aa, alphabet)

        # backprop to get gradients
        score.backward()

        # record: gradient × (mut - wt) for each feature
        gradient = interpolated.grad  # (4096,)
        delta = mut_pre_acts.to(device) - wt_pre_acts.to(device)
        effect = gradient * delta

        effects.append(effect.detach().cpu())

    # average effects across all integration steps
    effects_sae = torch.stack(effects).mean(dim=0)  # (4096,)

    # collect diagnostic info
    info = {
        'baseline_score': baseline_score.item(),
        'wt_pre_acts': wt_pre_acts.cpu().numpy(),
        'mut_pre_acts': mut_pre_acts.cpu().numpy(),
        'feature_deltas': (mut_pre_acts - wt_pre_acts).cpu().numpy(),
    }

    return effects_sae, info


def topk_features(effects_sae: torch.Tensor, k: int = 20, mode: str = 'abs') -> dict:
    """
    extract top-k most important features from IG attributions

    args:
        effects_sae: (4096,) feature attributions
        k: number of top features to return
        mode: 'abs' (by magnitude), 'pos' (positive only), 'neg' (negative only)

    returns:
        dict with top feature indices, values, and statistics
    """
    if mode == 'abs':
        ranking_vals = effects_sae.abs()
        top_vals, top_idx = torch.topk(ranking_vals, k, largest=True)
        # get original signed values
        top_effects = effects_sae[top_idx]
    elif mode == 'pos':
        top_effects, top_idx = torch.topk(effects_sae, k, largest=True)
    elif mode == 'neg':
        top_effects, top_idx = torch.topk(effects_sae, k, largest=False)
    else:
        raise ValueError(f"mode must be 'abs', 'pos', or 'neg', got {mode}")

    return {
        'indices': top_idx.cpu().numpy(),
        'effects': top_effects.cpu().numpy(),
        'total_effect': effects_sae.sum().item(),
        'top_k_fraction': top_effects.abs().sum().item() / effects_sae.abs().sum().item()
    }
