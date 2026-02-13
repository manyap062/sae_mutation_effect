"""analyze activation patching results"""
import os
import sys
import pandas as pd
import numpy as np


def load_results(path):
    """
    load one or more *_patching*.csv files
    if path is a directory: load all matching files
    if path is a file: load just that file
    """
    dfs = []

    if os.path.isfile(path):
        dfs.append(pd.read_csv(path))
    elif os.path.isdir(path):
        for f in os.listdir(path):
            if "_patching" in f and f.endswith(".csv"):
                dfs.append(pd.read_csv(os.path.join(path, f)))
    else:
        raise ValueError(f"{path} is not a file or a directory")

    if not dfs:
        raise RuntimeError(f"no *_patching*.csv files found under {path}")

    df = pd.concat(dfs, ignore_index=True)
    return df


def summarize_by_protein_layer(df):
    """aggregate core metrics per (protein, layer)"""
    grouped = df.groupby(["protein", "layer"], as_index=False)

    basic = grouped.agg(
        n_rows=("score_change", "size"),
        n_mutations=("mutation", "nunique"),
        mean_score_change=("score_change", "mean"),
        median_score_change=("score_change", "median"),
        mean_abs_score_change=("score_change", lambda x: x.abs().mean()),
        median_abs_score_change=("score_change", lambda x: x.abs().median()),
        mean_abs_feature_delta=("feature_delta", lambda x: x.abs().mean()),
    )

    # correlation between |feature_delta| and |score_change|
    corr_records = []
    for (protein, layer), g in df.groupby(["protein", "layer"]):
        if g.shape[0] < 2:
            corr = np.nan
        else:
            corr = np.corrcoef(g["feature_delta"].abs(), g["score_change"].abs())[0, 1]
        corr_records.append(
            {"protein": protein, "layer": layer, "abs_delta_score_corr": corr}
        )
    corr_df = pd.DataFrame(corr_records)

    summary = basic.merge(corr_df, on=["protein", "layer"], how="left")
    return summary


def summarize_by_mutation(df):
    """
    aggregate metrics per (protein, layer, mutation)
    each row corresponds to one mutation and its top-k patched features
    """
    grouped = df.groupby(["protein", "layer", "mutation"], as_index=False)

    summary = grouped.agg(
        position=("position", "first"),
        n_features=("feature_idx", "size"),
        mean_score_change=("score_change", "mean"),
        mean_abs_score_change=("score_change", lambda x: x.abs().mean()),
        mean_abs_feature_delta=("feature_delta", lambda x: x.abs().mean()),
    )
    return summary


def main(path):
    df = load_results(path)

    print("=== raw patched feature-level results (first 20 rows) ===")
    print(df.head(20).to_string())
    print()

    summary_pl = summarize_by_protein_layer(df)
    summary_mut = summarize_by_mutation(df)

    print("=== summary by (protein, layer) ===")
    print(summary_pl.to_string(index=False))
    print()

    print("=== summary by (protein, layer, mutation) ===")
    print(summary_mut.to_string(index=False))
    print()

    # save summaries next to the input
    if os.path.isdir(path):
        out_dir = path
    else:
        out_dir = os.path.dirname(path)

    os.makedirs(out_dir, exist_ok=True)

    summary_pl_path = os.path.join(out_dir, "patching_summary_protein_layer.csv")
    summary_mut_path = os.path.join(out_dir, "patching_summary_mutation.csv")

    summary_pl.to_csv(summary_pl_path, index=False)
    summary_mut.to_csv(summary_mut_path, index=False)

    print(f"saved protein/layer summary to: {summary_pl_path}")
    print(f"saved mutation-level summary to: {summary_mut_path}")


if __name__ == "__main__":
    PROJECT_DIR = "/project/pi_annagreen_umass_edu/manya/sae_mutation_effect"
    results_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else f"{PROJECT_DIR}/results/patching"
    )
    main(results_path)
