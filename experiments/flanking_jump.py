"""
Test if ESM-2 fitness prediction has a jump phenomenon under increasing levels context masking for a given mutation position

Mutation position always masked, for each radius r the residues within r of the target are also masked, rest of sequence unperturbed
At r=0 only the target is masked (full flanking context)
A sharp drop in log P(wt_aa) at some r = residues are critical context

python experiments/flanking_jump.py \
    --protein DNJA1_HUMAN \
    --position 7 \
    --esm_model_path /datasets/bio/esm/models/esm2_t33_650M_UR50D.pt \
    --device cpu

Outputs:
    - stdout table: per-radius log P(Leu), top 3 AAs, LLR for test mutations
    - flanking_jump_L7.png: log P(Leu) vs flanking radius
"""

import os
import sys
import argparse

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from src.esm_utils import load_esm_local
from src.data_utils import PROTEINS

#test_muts = ["L7A", "L7P", "L7D"]
test_muts = ["K33D", "K33E", "K33G"]


def seq_to_tokens(seq_list, alphabet, device):
    """
    seq_list: list of str or None, None means <mask>
    returns (1, L+2) token tensor
    """
    mask_idx = alphabet.mask_idx
    bos_idx = alphabet.cls_idx
    eos_idx = alphabet.eos_idx

    tokens = [bos_idx]
    for aa in seq_list:
        tokens.append(mask_idx if aa is None else alphabet.get_idx(aa))
    tokens.append(eos_idx)
    return torch.tensor([tokens], dtype=torch.long, device=device)


def get_log_probs_at(model, tokens, tok_pos):
    """Forward pass, log-softmax probs at tok_pos (1-indexed token position)"""
    with torch.no_grad():
        results = model(tokens, repr_layers=[], return_contacts=False)
    logits = results["logits"]  # (1, L+2, vocab)
    log_probs = torch.log_softmax(logits[0, tok_pos], dim=-1)
    return log_probs  # (vocab,)


def run_experiment(model, alphabet, wt_seq, target_pos_0, radii, test_muts, device):
    """
    for each radius, build masked token sequence, run forward pass, extract log P(aa) at the masked target position.

    returns list of dicts with keys: radius, log_p_wt, top3, llr_per_mut
    """
    target_pos_1 = target_pos_0 + 1   # 1-indexed (used for token index, BOS at 0)
    tok_pos = target_pos_1             # token index in (BOS, seq.. , EOS)
    wt_aa = wt_seq[target_pos_0]

    wt_idx = alphabet.get_idx(wt_aa)

    parsed_muts = []
    for mut_str in test_muts:
        # wt_m = mut_str[0]
        # pos_m = int(mut_str[1:-1])
        mut_aa = mut_str[-1]
        # assert wt_m == wt_aa, f"Mutation {mut_str} WT doesn't match sequence ({wt_aa})"
        # assert pos_m == target_pos_0 + 1, f"Mutation {mut_str} position != target"
        parsed_muts.append((mut_str, alphabet.get_idx(mut_aa)))

    rows = []
    for r in radii:
        label = "all" if r == -1 else str(r)

        seq_list = list(wt_seq)
        L = len(seq_list)
        for i in range(L):
            if r == -1:
                # no context: mask everything
                seq_list[i] = None
            else:
                # mask target + all residues within radius r
                if abs(i - target_pos_0) <= r:
                    seq_list[i] = None

        print(f"{r}: {seq_list}")

        tokens = seq_to_tokens(seq_list, alphabet, device)
        log_probs = get_log_probs_at(model, tokens, tok_pos)

        log_p_wt = log_probs[wt_idx].item()

        # top 3 AAs
        aa_lp = [(alphabet.get_tok(i), log_probs[i].item())
                 for i in range(len(alphabet))]
        aa_lp.sort(key=lambda x: -x[1])
        top3 = aa_lp[:3]

        # LLR for each test mutation
        llr_per_mut = {}
        for mut_str, mut_idx in parsed_muts:
            llr = (log_probs[mut_idx] - log_probs[wt_idx]).item()
            llr_per_mut[mut_str] = llr

        rows.append({
            "radius": label,
            "log_p_wt": log_p_wt,
            "top3": top3,
            "llr_per_mut": llr_per_mut,
        })

    return rows, wt_aa


def print_results(rows, wt_aa, target_pos_1):
    print(f"\n Flanking jump experiment,  position {target_pos_1} ({wt_aa}) \n")
    mut_keys = list(rows[0]["llr_per_mut"].keys())

    header = f"{'radius':>8}  {'log P('+wt_aa+')':>12}  {'top-3 (aa, logP)':^42}"
    for k in mut_keys:
        header += f"  {'LLR('+k+')':>12}"
    print(header)
    print("-" * len(header))

    for row in rows:
        top3_str = "  ".join(f"{aa}:{lp:+.3f}" for aa, lp in row["top3"])
        line = f"{row['radius']:>8}  {row['log_p_wt']:>+12.4f}  {top3_str:<42}"
        for k in mut_keys:
            line += f"  {row['llr_per_mut'][k]:>+12.4f}"
        print(line)
    print()


def plot_results(rows, wt_aa, target_pos_1, out_path):
    # x-axis: numeric radii 
    x_labels = [r["radius"] for r in rows]
    x_numeric = list(range(len(x_labels)))
    y = [r["log_p_wt"] for r in rows]

    # baseline (r=0): only mut pos masked
    baseline_val = y[0]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x_numeric, y, marker="o", linewidth=2, markersize=7, label=f"log P({wt_aa} | context)")
    ax.axhline(baseline_val, color="red", linestyle="--", linewidth=1.5, label="r=0 baseline (full context)")

    ax.set_xticks(x_numeric)
    ax.set_xticklabels(x_labels, rotation=30, ha="right")
    ax.set_xlabel("Radius (residues masked on each side of mutation)")
    ax.set_ylabel(f"log P({wt_aa} | context)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Plot saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Flanking jump experiment for ESM-2")
    parser.add_argument("--protein", default="DNJA1_HUMAN", choices=list(PROTEINS.keys()))
    parser.add_argument("--position", type=int, default=7,
                        help="1-indexed target position (default: 7 = L7)")
    parser.add_argument("--esm_model_path",
                        default="/datasets/bio/esm/models/esm2_t33_650M_UR50D.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--test_muts", nargs="*", default=test_muts,
                        help="Substitutions to track LLR for (L7A L7P L7D)")
    parser.add_argument("--out", default="flanking_jump.png",
                        help="Output plot")
    args = parser.parse_args()

    device = torch.device(args.device)
    wt_seq = PROTEINS[args.protein]["wt_seq"]
    target_pos_0 = args.position - 1  # 0 indexed

    print(f"Protein:  {args.protein}  (len={len(wt_seq)})")
    print(f"Sequence: {wt_seq}")
    print(f"Target:   position {args.position} = {wt_seq[target_pos_0]}")
    print(f"Loading ESM-2 from {args.esm_model_path} ...")

    model, alphabet, _ = load_esm_local(args.esm_model_path, device)


    # 0 = only target masked (baseline), -1 = mask everything (no context)
    radii = [0, 1, 6, 8, 9, 11, 13, 16, 19, 20, 21, 22, 23, 24, 25, -1]
    #radii = [30, 31, 32, 33, 34, 35, 36]
    print(f"Running {len(radii)} forward passes ...\n")
    rows, wt_aa = run_experiment(
        model, alphabet, wt_seq, target_pos_0, radii, args.test_muts, device
    )

    print_results(rows, wt_aa, args.position)
    plot_results(rows, wt_aa, args.position, args.out)


if __name__ == "__main__":
    main()
