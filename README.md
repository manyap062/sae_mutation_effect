# SAE mutation effect prediction

Minimal circuit discovery in ESM-2-650M using sparse autoencoders. Aim to find which SAE features causally mediate mutation effect prediction.

## Setup

Runs on unity HPCC cluster

```bash
source ~/esm_env/bin/activate

# create environment:
python3 -m venv esm_env
source esm_env/bin/activate
pip install torch numpy pandas safetensors fair-esm
```

**Required files**:
- **ESM-2 checkpoint**: Download ESM2-650M from [facebookresearch/esm](https://github.com/facebookresearch/esm)
  - On Unity cluster: `/datasets/bio/esm/models/esm2_t33_650M_UR50D.pt`
  - Specify custom path: `--esm_model_path /path/to/esm2_t33_650M_UR50D.pt`
- **SAE weights**: Place in `sae_weights/` directory (Adams et al. TopK SAEs)
  - Format: `esm2_plm1280_l{layer}_sae4096.safetensors` for each layer
  - Specify custom directory: `--sae_weights_dir /path/to/weights/`
- **Mutation data**: Included in repo at `data/*.csv` (high-impact mutations pre-filtered)

## Quick start

Run activation patching experiments:
```bash
python experiments/run_patching.py --protein EPHB2_HUMAN --layer 16 --n_mutations 100
```

Run integrated gradients:
```bash
python experiments/run_ig.py --protein EPHB2_HUMAN --layers 16 --n_mutations 100
```

**Custom paths** (if not using defaults):
```bash
python experiments/run_patching.py \
  --protein EPHB2_HUMAN \
  --layer 16 \
  --esm_model_path /path/to/esm2_t33_650M_UR50D.pt \
  --sae_weights_dir /path/to/sae_weights/ \
  --output_dir /path/to/output/
```

Submit slurm jobs:
```bash
sbatch scripts/submit_patching.sh
sbatch scripts/submit_ig.sh
```

## Structure

- `src/` - library code (SAE, scoring, patching, IG)
- `experiments/` - slurm scripts
- `analysis/` - analysis + figures
- `results/` - outputs
- `data/` - protein sequences + mutations

## proteins

- EPHB2_HUMAN (68aa, ~50 high-impact mutations)
- DNJA1_HUMAN (66aa, ~50 high-impact mutations)
- PR40A_HUMAN (64aa, ~50 high-impact mutations)


## key findings

- Peak causal effects at layer 16
- Extreme sparsity: top 20 features = 95-98% of IG attribution
- Single latent features cause 5-15 log-odds changes
- Universal features across proteins (AA detectors, for ex. leucine detectors 2339, 1237)

## Important methodological notes

- WT marginal scoring (not using masked-marginal, masked tokens degrades SAE performance)
- Activation patching 
- Baseline and patched paths must use identical sae encode/decode
- No-op controls should be exactly 0.00

## Files

### src/
- `sae.py` - SparseAutoencoder class (d_model=1280, d_hidden=4096, k=32)
- `scoring.py` - WT marginal scoring
- `patching.py` - activation patching logic
- `ig.py` - integrated gradients
- `data_utils.py` - mutation loading
- `esm_utils.py` - ESM-2 model loading

### experiments/
- `run_patching.py` - activation patching experiments
- `run_ig.py` - integrated gradients experiments

### results/
- `patching/` - patching csvs (3 proteins × 5 layers)
- `ig/` - IG .npz files (~2400 files)
- `figures/` - graphs


