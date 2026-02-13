"""mutation data loading"""
import os
import pandas as pd


# protein sequences and high-impact csv files
# csvs are already filtered for high impact mutations
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_PROJECT_DIR, "data")

PROTEINS = {
    "EPHB2_HUMAN": {
        "wt_seq": "SFNTVDEWLEAIKMGQYKESFANAGFTSFDVVSQMMMEDILRVGVTLAGHQKKILNSIQVMRAQMN",
        "csv": os.path.join(_DATA_DIR, "EPHB2_HUMAN_high_impact.csv")
    },
    "DNJA1_HUMAN": {
        "wt_seq": "TTYYDVLGVKPNATQEELKKAYRKLALKYHPDKNPNEGEKFKQISQAYEVLSDAKKRELYDKGGE",
        "csv": os.path.join(_DATA_DIR, "DNJA1_HUMAN_high_impact.csv")
    },
    "PR40A_HUMAN": {
        "wt_seq": "TYTWNTKEEAKQAFKELLKEKRVPSNASWEQAMKMIINDPRYSALAKLSEKKQAFNAYKVQTE",
        "csv": os.path.join(_DATA_DIR, "PR40A_HUMAN_high_impact_FIXED.csv")
    }
}


def parse_mutation_string(mut_str):
    """
    parse mutation string to (wt, pos, mut)
    format: "L7A" → (wt='L', pos=7, mut='A')
    """
    wt = mut_str[0]
    mut = mut_str[-1]
    pos = int(mut_str[1:-1])
    return wt, pos, mut


def load_mutations(protein_name, n_mutations=None, seed=42):
    """
    load mutations for a protein from high impact csv
    csvs are already filtered for high impact, no additional filtering needed
    samples n_mutations randomly if specified and more are available
    """
    csv_path = PROTEINS[protein_name]["csv"]
    df = pd.read_csv(csv_path)

    if n_mutations is None or len(df) <= n_mutations:
        print(f"using {len(df)} mutations for {protein_name}")
        df_selected = df
    else:
        df_selected = df.sample(n=n_mutations, random_state=seed)
        print(f"using {n_mutations}/{len(df)} possible mutations for {protein_name}")

    return df_selected
