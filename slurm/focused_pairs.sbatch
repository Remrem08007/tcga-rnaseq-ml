#!/usr/bin/env bash
#SBATCH --job-name=tcga-pairs
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/tcga-pairs-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
MATRIX="${MATRIX:-$REPO_ROOT/data/cache/expression.float32.npy}"
SPLIT="${SPLIT:-$REPO_ROOT/data/processed/split_manifest.tsv}"
GENES="${GENES:-$REPO_ROOT/data/cache/genes.tsv}"
OUTDIR="${OUTDIR:-$REPO_ROOT/results/focused_pairs/elastic_net}"
MODEL="${MODEL:-elastic_net}"
GENE_BUDGET="${GENE_BUDGET:-1000}"
NEGATIVE_POLICY="${NEGATIVE_POLICY:-clip}"
N_JOBS="${N_JOBS:-${SLURM_CPUS_PER_TASK:-1}}"

module --force purge
module load StdEnv/2023
module load python/3.11

cd "$REPO_ROOT"
source "$REPO_ROOT/.venv/bin/activate"
mkdir -p "$REPO_ROOT/logs" "$OUTDIR"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

if [[ -s "$OUTDIR/focused_pairs.json" ]]; then
  echo "[focused-pairs] complete output already exists: $OUTDIR/focused_pairs.json"
  exit 0
fi

python -u -m tcga_ml.focused_pairs_cli \
  --matrix "$MATRIX" \
  --split "$SPLIT" \
  --genes "$GENES" \
  --outdir "$OUTDIR" \
  --model "$MODEL" \
  --gene-budget "$GENE_BUDGET" \
  --negative-policy "$NEGATIVE_POLICY" \
  --n-jobs "$N_JOBS"

test -s "$OUTDIR/focused_pairs.json"
test -s "$OUTDIR/focused_pair_predictions.tsv"
echo "[focused-pairs] complete: $OUTDIR/focused_pairs.json"
