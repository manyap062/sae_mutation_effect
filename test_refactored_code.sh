#!/bin/bash


set -e  # exit on error

PROJECT_DIR="/project/pi_annagreen_umass_edu/manya/sae_mutation_effect"
TEST_DIR="${PROJECT_DIR}/new_code_test"

echo "Testing Refactored Code"
echo ""

rm -rf "${TEST_DIR}"
mkdir -p "${TEST_DIR}/patching"
mkdir -p "${TEST_DIR}/ig"

cd "${PROJECT_DIR}"

# activate environment
source ~/esm_env/bin/activate

# add project directory to python path so it can find src/
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

# test parameters
PROTEIN="EPHB2_HUMAN"
N_MUTATIONS=3
LAYERS="8 16 24"  # test multiple layers to check peak
SEED=42

echo "Test parameters:"
echo " Protein: ${PROTEIN}"
echo " Mutations: ${N_MUTATIONS}"
echo " Layers: ${LAYERS}"
echo " Seed: ${SEED}"
echo ""

# 1. PATCHING (with controls)

echo "1. Testing Patching (with controls)"

for LAYER in $LAYERS; do
    echo ""
    echo "Layer ${LAYER}:"
    python experiments/run_patching.py \
        --protein ${PROTEIN} \
        --layer ${LAYER} \
        --n_mutations ${N_MUTATIONS} \
        --control_types real noop random \
        --output_dir ${TEST_DIR}/patching \
        --seed ${SEED} \
        --device cpu
done

echo "Done"

# 2. TEST IG

echo "2. Testing Integrated Gradients"

python experiments/run_ig.py \
    --protein ${PROTEIN} \
    --layers "8,16,24" \
    --n_mutations ${N_MUTATIONS} \
    --output_dir ${TEST_DIR}/ig \
    --seed ${SEED} \
    --ig_steps 10 \
    --device cpu

echo "Done"
echo ""

# 3. TEST ANALYSIS


python analysis/analyze_patching.py ${TEST_DIR}/patching > ${TEST_DIR}/patching_analysis.txt
echo "Patching analysis done"
echo ""

python analysis/analyze_ig.py ${TEST_DIR}/ig > ${TEST_DIR}/ig_analysis.txt
echo "IG analysis done"
echo ""

echo "4. Checks"

python << 'PYEOF'
import pandas as pd
import numpy as np
import os

TEST_DIR = "new_code_test"

print("=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)
print()

print("1. No-op Control")
patching_files = [f for f in os.listdir(f"{TEST_DIR}/patching") if f.endswith('.csv') and 'summary' not in f]
for f in patching_files:
    df = pd.read_csv(f"{TEST_DIR}/patching/{f}")
    noop = df[df['control_type'] == 'noop']
    if len(noop) > 0:
        mean_noop = noop['abs_score_change'].mean()
        max_noop = noop['abs_score_change'].max()
        layer = noop['layer'].iloc[0]
        print(f"  Layer {layer}: mean={mean_noop:.6f}, max={max_noop:.6f} {status}")

print()

print("2. Real vs Random (real should be > random)")
for f in patching_files:
    df = pd.read_csv(f"{TEST_DIR}/patching/{f}")
    real = df[df['control_type'] == 'real']['abs_score_change'].mean()
    random = df[df['control_type'] == 'random']['abs_score_change'].mean()
    if real > 0 and random >= 0:
        ratio = real / random if random > 0 else float('inf')
        layer = df['layer'].iloc[0]
        print(f"  Layer {layer}: real={real:.4f}, random={random:.6f}, ratio={ratio:.0f}x {status}")

print()

print("3. Layer Progression (peak around layer 16)")
layer_effects = {}
for f in patching_files:
    df = pd.read_csv(f"{TEST_DIR}/patching/{f}")
    real = df[df['control_type'] == 'real']
    if len(real) > 0:
        layer = real['layer'].iloc[0]
        layer_effects[layer] = real['abs_score_change'].mean()

for layer in sorted(layer_effects.keys()):
    print(f"  Layer {layer}: mean effect = {layer_effects[layer]:.4f}")

if len(layer_effects) > 1:
    peak_layer = max(layer_effects, key=layer_effects.get)
    print(f"  Peak layer: {peak_layer}")

print()

print("4. IG Sparsity")
print("-" * 60)
ig_files = [f for f in os.listdir(f"{TEST_DIR}/ig") if f.endswith('_ig.npz')]
top20_fractions = []
for f in ig_files:
    data = np.load(f"{TEST_DIR}/ig/{f}", allow_pickle=True)
    if 'top_20_fraction' in data:
        frac = float(data['top_20_fraction'])
        top20_fractions.append(frac)

if top20_fractions:
    mean_frac = np.mean(top20_fractions)
    print(f"  Mean top-20 fraction: {mean_frac:.4f} {status}")
    print(f"  (top 20 features = {mean_frac*100:.1f}% of total attribution)")

print()


PYEOF

echo ""

#####################################
# 5. SUMMARY
#####################################

echo "Output directory: ${TEST_DIR}"
echo ""
echo "Files created:"
echo "  Patching:"
ls -lh ${TEST_DIR}/patching/*.csv 2>/dev/null | head -10
echo ""
echo "  IG:"
ls -lh ${TEST_DIR}/ig/*.npz 2>/dev/null | head -10
echo ""
echo "Analysis outputs:"
cat ${TEST_DIR}/patching_analysis.txt | head -30
echo ""

