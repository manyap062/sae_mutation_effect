#!/bin/bash
# submit all experiments at once

echo "submitting patching experiments..."
sbatch slurm/submit_patching.sh

echo "submitting IG experiments..."
sbatch slurm/submit_ig.sh

echo "all jobs submitted!"
