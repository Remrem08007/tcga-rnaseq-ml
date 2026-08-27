#!/usr/bin/env bash
#SBATCH --job-name=tcga-xgb-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/tcga-xgb-cpu-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
MATRIX="${MATRIX:-$REPO_ROOT/data/cache/expression.float32.npy}"
SPLIT="${SPLIT:-$REPO_ROOT/data/processed/split_manifest.tsv}"
OUTDIR="${OUTDIR:-$REPO_ROOT/results/compute/xgboost-cpu}"
GENE_BUDGET="${GENE_BUDGET:-1000}"
NEGATIVE_POLICY="${NEGATIVE_POLICY:-clip}"
THREAD_LIMIT="${SLURM_CPUS_PER_TASK:-16}"

mkdir -p "$REPO_ROOT/logs" "$OUTDIR"
cd "$REPO_ROOT"

CPU_ARGS=()
for THREAD_COUNT in 1 2 4 8 16; do
  if (( THREAD_COUNT <= THREAD_LIMIT )); then
    CPU_ARGS+=(--cpu-threads "$THREAD_COUNT")
  fi
done

python -m tcga_ml.xgboost_cli scale \
  --matrix "$MATRIX" \
  --split "$SPLIT" \
  --outdir "$OUTDIR" \
  "${CPU_ARGS[@]}" \
  --gene-budget "$GENE_BUDGET" \
  --negative-policy "$NEGATIVE_POLICY"
