#!/bin/bash
#SBATCH --job-name=sae_ig
#SBATCH --output=results/ig/logs/%A_%a.out
#SBATCH --error=results/ig/logs/%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-2

# activate environment
source ~/esm_env/bin/activate

# project directory
PROJECT_DIR="/project/pi_annagreen_umass_edu/manya/sae_mutation_effect"
cd "$PROJECT_DIR"

# proteins
PROTEINS=("EPHB2_HUMAN" "DNJA1_HUMAN" "PR40A_HUMAN")
PROTEIN=${PROTEINS[$SLURM_ARRAY_TASK_ID]}

echo "running IG for ${PROTEIN}"

# run experiment (all layers for one protein)
python experiments/run_ig.py \
    --protein ${PROTEIN} \
    --layers 4,8,12,16,20,24,28,32 \
    --output_dir results/ig \
    --n_mutations 100 \
    --ig_steps 10 \
    --device cpu \
    --seed 42

echo "done!"
