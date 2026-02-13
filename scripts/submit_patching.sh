#!/bin/bash
#SBATCH --job-name=sae_patch
#SBATCH --output=results/patching/logs/%A_%a.out
#SBATCH --error=results/patching/logs/%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-14

# activate environment
source ~/esm_env/bin/activate

# project directory
PROJECT_DIR="/project/pi_annagreen_umass_edu/manya/sae_mutation_effect"
cd "$PROJECT_DIR"

# proteins, layers, and control types
PROTEINS=("EPHB2_HUMAN" "DNJA1_HUMAN" "PR40A_HUMAN")
LAYERS=(8 12 16 20 24)

# compute array indices → (protein, layer)
N_LAYERS=${#LAYERS[@]}
PROTEIN_IDX=$((SLURM_ARRAY_TASK_ID / N_LAYERS))
LAYER_IDX=$((SLURM_ARRAY_TASK_ID % N_LAYERS))

PROTEIN=${PROTEINS[$PROTEIN_IDX]}
LAYER=${LAYERS[$LAYER_IDX]}

echo "running patching for ${PROTEIN} at layer ${LAYER}"

# run experiment
python experiments/run_patching.py \
    --protein ${PROTEIN} \
    --layer ${LAYER} \
    --output_dir results/patching \
    --n_mutations 100 \
    --device cpu \
    --seed 42

echo "done!"
