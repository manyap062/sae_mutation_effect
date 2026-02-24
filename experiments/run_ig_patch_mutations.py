"""
IG + activation patching

For each mutation (V6P):
  1. Run integrated_gradients_mutation() at the target layer
     (or load from cached .npz if already computed w/ run_ig.py)
  2. Extract top-k latents by absolute IG attribution
  3. For each latent, use activation patching (WT pre-act to mut pre-act and measure the change in wt_marginal_score vs baseline

  python experiments/run_ig_patch_mutations.py \
      --protein DNJA1_HUMAN \
      --mutations V6P L7D V9D \
      --layer 16 \
      --top_k 5
"""

"TODO: Find topK according to x% performance recovery"

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from src import (
    PROTEINS,
    load_esm_local, load_sae, tokenize_seq, get_logits_and_hidden,
    integrated_gradients_mutation, topk_features,
    wt_marginal_score, forward_from_layer, get_sae_activations,
    parse_mutation_string,
)


def make_mutant_seq(wt_seq: str, pos: int, mut_aa: str) -> str:
    """Substitute single residue at 1-indexed position"""
    seq = list(wt_seq)
    seq[pos - 1] = mut_aa
    return "".join(seq)


def load_ig_cache(protein, mutation_str, layer, ig_dir):
    """
    Load IG attributions from results/ig/ if available
    Returns (effects_sae tensor, baseline_score float) or (None, None)
    """
    fname = f"{protein}_{mutation_str.replace('>', '_')}_layer{layer}_ig.npz"
    fpath = os.path.join(ig_dir, fname)
    if os.path.exists(fpath):
        d = np.load(fpath, allow_pickle=True)
        effects_sae = torch.tensor(d["all_effects"])  # (4096,)
        baseline_score = float(d["baseline_score"])
        return effects_sae, baseline_score
    return None, None


def run_one_mutation(
    esm_model, sae, alphabet, batch_converter,
    wt_seq, protein, mutation_str, layer, top_k, ig_steps, device, ig_dir
):
    """
    Run IG (or load cache) then patch top-k latents for one mutation at one layer.
    Returns list of dicts (one per top latent).
    """
    wt_aa, pos, mut_aa = parse_mutation_string(mutation_str)
    mut_seq = make_mutant_seq(wt_seq, pos, mut_aa)

    wt_tokens = tokenize_seq(wt_seq, batch_converter, device)
    mut_tokens = tokenize_seq(mut_seq, batch_converter, device)

    # load cached IG results if available, otherwise recompute
    effects_sae, baseline_score = load_ig_cache(protein, mutation_str, layer, ig_dir)
    if effects_sae is not None:
        print(f"  loaded IG from cache for {mutation_str} layer {layer}")
    else:
        print(f"  running IG for {mutation_str} at layer {layer} ({ig_steps} steps)")
        effects_sae, info = integrated_gradients_mutation(
            esm_model=esm_model,
            sae_model=sae,
            wt_tokens=wt_tokens,
            mut_tokens=mut_tokens,
            position=pos,
            wt_aa=wt_aa,
            mut_aa=mut_aa,
            alphabet=alphabet,
            hook_layer=layer,
            steps=ig_steps,
            device=device,
        )
        baseline_score = info["baseline_score"]

    # top k latents by most negative IG attribution
    top = topk_features(effects_sae, k=top_k, mode="neg")
    top_indices = top["indices"]   # (k,) numpy int array
    top_effects = top["effects"]   # (k,) signed float array

    # activation patching: need WT/mut hidden states and SAE pre-acts
    _, wt_hidden = get_logits_and_hidden(esm_model, wt_tokens, layer)
    _, mut_hidden = get_logits_and_hidden(esm_model, mut_tokens, layer)

    tok_pos = pos  # 1-indexed == token position (BOS at 0)
    wt_pre_acts, wt_mu, wt_std = get_sae_activations(sae, wt_hidden, tok_pos)
    mut_pre_acts, mut_mu,mut_std = get_sae_activations(sae, mut_hidden, tok_pos)

    rows = []
    for rank, (feat_idx, ig_attr) in enumerate(zip(top_indices, top_effects), start=1):
        feat_idx = int(feat_idx)

        # patch only this one latent from WT to mutant pre-activation value
        # patch_val = mut_pre_acts[feat_idx].reshape(1, 1)  # (1,1)
        # patched_h = sae.decode_with_patch(
        #     wt_pre_acts.unsqueeze(0), wt_mu, wt_std,
        #     patch_indices=torch.tensor([feat_idx], device=device),
        #     patch_values=patch_val,
        # ).squeeze(0)

        # wt_hidden_patched = wt_hidden.clone()
        # wt_hidden_patched[0, tok_pos] = patched_h

        # patched_logits = forward_from_layer(esm_model, wt_hidden_patched, layer)
        # patched_score = wt_marginal_score(patched_logits, pos, wt_aa, mut_aa, alphabet)


        # NOTE: patch only this one latent from mutant to wild type pre-activation value to see how much the WT effect can be recovered by just fixing this one latent
        patch_val = wt_pre_acts[feat_idx].reshape(1, 1)  # (1,1)
        patched_h = sae.decode_with_patch(
            mut_pre_acts.unsqueeze(0), mut_mu, mut_std,
            patch_indices=torch.tensor([feat_idx], device=device),
            patch_values=patch_val,
        ).squeeze(0)

        mut_hidden_patched = mut_hidden.clone()
        mut_hidden_patched[0, tok_pos] = patched_h

        patched_logits = forward_from_layer(esm_model, mut_hidden_patched, layer)
        s_patched = wt_marginal_score(patched_logits, pos, wt_aa, mut_aa, alphabet)

        recovery_denominator = info["s_wt"] - info["s_mut"]
        recovery_numerator = s_patched - info["s_mut"]

        # NOTE: Handle 0 denominator error later
        recovery = recovery_numerator / recovery_denominator
        # wt_marginal_score returns a plain float

        rows.append({
            "position":             pos,
            "mutation":             mutation_str,
            "latent_idx":           feat_idx,
            "ig_attribution":       float(ig_attr),
            # "patched_score_change": patched_score - baseline_score,
            "recovery":            recovery,
            "baseline_score":       baseline_score,
            "rank":                 rank,
        })

    return rows


def main():
    parser = argparse.ArgumentParser(description="IG + activation patching for provided mutations")
    parser.add_argument("--protein", required=True,
                        choices=list(PROTEINS.keys()),
                        help="protein name, e.g. DNJA1_HUMAN")
    parser.add_argument("--mutations", nargs="+", required=True,
                        help="mutation strings, ex. V6P L7D V9D")
    parser.add_argument("--layer", type=int, default=16,
                        help="ESM layer to hook at (default: 16)")
    parser.add_argument("--top_k", type=int, default=5,
                        help="number of top latents to extract and patch (default= 5)")
    parser.add_argument("--ig_steps", type=int, default=10,
                        help="IG integration steps (default= 10)")
    parser.add_argument("--output", default=None,
                        help="output CSV path (default: PROJECT_DIR/results/ig_top5_core_positions.csv)")
    parser.add_argument("--esm_model_path",
                        default="/datasets/bio/esm/models/esm2_t33_650M_UR50D.pt")
    parser.add_argument("--sae_weights_dir", default=None,
                        help="SAE weights dir (default: PROJECT_DIR/sae_weights)")
    parser.add_argument("--ig_dir", default=None,
                        help="dir to look for cached IG .npz files (default: PROJECT_DIR/results/ig)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(PROJECT_DIR, "results", "ig_top5_core_positions.csv")
    if args.sae_weights_dir is None:
        args.sae_weights_dir = os.path.join(PROJECT_DIR, "sae_weights")
    if args.ig_dir is None:
        args.ig_dir = os.path.join(PROJECT_DIR, "results", "ig")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    device = torch.device(args.device)

    print(f"protein:    {args.protein}")
    print(f"mutations:  {args.mutations}")
    print(f"layer:      {args.layer}")

    wt_seq = PROTEINS[args.protein]["wt_seq"]

    print(f"Loading ESM-2 from {args.esm_model_path}")
    esm_model, alphabet, batch_converter = load_esm_local(args.esm_model_path, device)

    print(f"Loading SAE for layer {args.layer} from {args.sae_weights_dir}")
    sae = load_sae(args.layer, args.sae_weights_dir, device)

    all_rows = []
    for mut_str in args.mutations:
        print(f"Mutation: {mut_str}")
        rows = run_one_mutation(
            esm_model, sae, alphabet, batch_converter,
            wt_seq, args.protein, mut_str, args.layer, args.top_k, args.ig_steps,
            device, args.ig_dir,
        )
        for r in rows:
            r["protein"] = args.protein
            r["layer"] = args.layer
        all_rows.extend(rows)

        for r in rows:
            print(f"  rank {r['rank']:>2}: latent {r['latent_idx']:>5}  "
                  f"ig={r['ig_attribution']:+.4f}  "
                  f"recovery={r['recovery']:+.4f}  "
                  f"baseline={r['baseline_score']:.4f}")

    cols = ["protein", "layer", "position", "mutation", "latent_idx",
            "ig_attribution", "recovery", "baseline_score", "rank"]
    df = pd.DataFrame(all_rows, columns=cols)
    df.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()