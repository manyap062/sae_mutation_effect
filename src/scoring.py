"""WT marginal scoring for mutation effect prediction"""
import torch


def wt_marginal_score(logits, position, wt_aa, mut_aa, alphabet):
    """
    compute WT marginal score: log p(mut_aa | x^wt) - log p(wt_aa | x^wt)

    WT marginal scoring, NOT masked-marginal (sae breaks masked scoring)

    args:
        logits: (1, L+2, vocab) model output
        position: 1-indexed mutation position (token index after BOS)
        wt_aa: wild-type amino acid (single letter)
        mut_aa: mutant amino acid (single letter)
        alphabet: ESM alphabet for AA→index conversion

    returns:
        scalar: mutation effect score (negative = deleterious)
    """
    tok_pos = position  # 1-indexed position in sequence = token index (BOS is 0)
    log_probs = torch.log_softmax(logits[0, tok_pos], dim=-1)

    wt_idx = alphabet.get_idx(wt_aa)
    mut_idx = alphabet.get_idx(mut_aa)

    return (log_probs[mut_idx] - log_probs[wt_idx]).item()
