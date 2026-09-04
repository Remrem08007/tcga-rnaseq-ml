#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONWARNINGS="error::FutureWarning"

python -m pytest -q -W error::FutureWarning
python -m compileall -q src
python -m tcga_ml.download --list >/dev/null
python scripts/run_xgboost.py --help >/dev/null
python scripts/run_focused_pairs.py --help >/dev/null
python scripts/run_final_evaluation.py --help >/dev/null
python scripts/compare_candidates.py --help >/dev/null
bash -n \
  slurm/classical_benchmark.sbatch \
  slurm/feature_budget.sbatch \
  slurm/focused_pairs.sbatch \
  slurm/xgboost_cpu_cv.sbatch \
  slurm/xgboost_cpu_scaling.sbatch \
  slurm/xgboost_gpu.sbatch

cohort_tmp="$(mktemp -d)"
cache_tmp="$(mktemp -d)"
split_tmp="$(mktemp -d)"
benchmark_tmp="$(mktemp -d)"
feature_tmp="$(mktemp -d)"
xgboost_tmp="$(mktemp -d)"
focused_tmp="$(mktemp -d)"
comparison_tmp="$(mktemp -d)"
final_tmp="$(mktemp -d)"
cleanup() {
  rm -rf "$cohort_tmp" "$cache_tmp" "$split_tmp" "$benchmark_tmp" "$feature_tmp" "$xgboost_tmp" "$focused_tmp" "$comparison_tmp" "$final_tmp"
}
trap cleanup EXIT

python -m tcga_ml.cohort_cli \
  --expression tests/fixtures/expression_small.tsv \
  --quality tests/fixtures/quality_small.tsv \
  --outdir "$cohort_tmp" >/dev/null

test -s "$cohort_tmp/cohort.tsv"
test -s "$cohort_tmp/cohort_summary.json"

python -m tcga_ml.cache_cli \
  --expression tests/fixtures/expression_cache_small.tsv \
  --cohort tests/fixtures/cohort_cache_small.tsv \
  --outdir "$cache_tmp" \
  --chunk-genes 2 >/dev/null

python -m tcga_ml.audit_cli \
  --matrix "$cache_tmp/expression.float32.npy" \
  --output "$cache_tmp/audit.json" >/dev/null

python - "$cache_tmp" <<'PY'
import json
import sys
from pathlib import Path
import numpy as np

root = Path(sys.argv[1])
matrix = np.load(root / "expression.float32.npy", mmap_mode="r")
assert matrix.shape == (2, 3)
assert matrix.dtype == np.float32
audit = json.loads((root / "audit.json").read_text())
assert audit["shape"] == [2, 3]
PY

python -m tcga_ml.split_cli \
  --samples tests/fixtures/samples_split_small.tsv \
  --outdir "$split_tmp" \
  --holdout-fraction 0.2 \
  --seed 20260825 >/dev/null

python - "$split_tmp" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
summary = json.loads((root / "split_summary.json").read_text())
assert summary["n_development"] == 12
assert summary["n_holdout"] == 3
assert len(summary["manifest_sha256"]) == 64
PY

python - "$benchmark_tmp" <<'PY'
import csv
import sys
from pathlib import Path
import numpy as np

root = Path(sys.argv[1])
rng = np.random.default_rng(3)
blocks = []
labels = []
for class_index, label in enumerate(["BRCA", "LUAD", "LUSC"]):
    block = rng.lognormal(0.5, 0.2, size=(15, 9)).astype("float32")
    block[:, class_index * 3 : class_index * 3 + 3] *= 6
    blocks.append(block)
    labels.extend([label] * 15)
np.save(root / "x.npy", np.vstack(blocks))
with (root / "split.tsv").open("w", newline="") as handle:
    writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
    writer.writerow(["cache_index", "participant_barcode", "cancer_type", "split"])
    for index, label in enumerate(labels):
        writer.writerow(
            [index, f"TCGA-AA-{index:04d}", label, "holdout" if index % 5 == 0 else "development"]
        )
with (root / "genes.tsv").open("w", newline="") as handle:
    writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
    writer.writerow(["gene_index", "source_gene_id", "symbol", "entrez_id"])
    for index in range(9):
        writer.writerow([index, f"G{index}|{index}", f"G{index}", index])
PY

python -m tcga_ml.benchmark_cli \
  --matrix "$benchmark_tmp/x.npy" \
  --split "$benchmark_tmp/split.tsv" \
  --outdir "$benchmark_tmp/out" \
  --model dummy \
  --model logistic_l2 \
  --cv-folds 3 \
  --n-jobs 1 \
  --no-progress >/dev/null

test -s "$benchmark_tmp/out/benchmark.json"

python -m tcga_ml.feature_budget_cli \
  --matrix "$benchmark_tmp/x.npy" \
  --split "$benchmark_tmp/split.tsv" \
  --genes "$benchmark_tmp/genes.tsv" \
  --outdir "$feature_tmp/out" \
  --model elastic_net \
  --gene-budget 2 \
  --gene-budget all \
  --cv-folds 3 \
  --n-jobs 1 \
  --no-progress >/dev/null

test -s "$feature_tmp/out/feature_budget.json"
test -s "$feature_tmp/out/feature_stability.tsv"
test -s "$feature_tmp/out/coefficients.tsv"

python -m tcga_ml.xgboost_cli probe >/dev/null
python -m tcga_ml.xgboost_cli cv \
  --matrix "$benchmark_tmp/x.npy" \
  --split "$benchmark_tmp/split.tsv" \
  --genes "$benchmark_tmp/genes.tsv" \
  --outdir "$xgboost_tmp/cv" \
  --device cpu \
  --threads 1 \
  --fold-jobs 1 \
  --gene-budget 4 \
  --cv-folds 2 \
  --n-estimators 4 \
  --max-depth 2 \
  --learning-rate 0.2 \
  --no-progress >/dev/null

test -s "$xgboost_tmp/cv/xgboost_benchmark.json"
test -s "$xgboost_tmp/cv/xgboost_feature_importance.tsv"

SLURM_CPUS_PER_TASK=2 python -m tcga_ml.xgboost_cli scale \
  --matrix "$benchmark_tmp/x.npy" \
  --split "$benchmark_tmp/split.tsv" \
  --outdir "$xgboost_tmp/scaling" \
  --cpu-threads 1 \
  --cpu-threads 2 \
  --gene-budget 4 \
  --cv-folds 2 \
  --n-estimators 3 \
  --max-depth 2 \
  --learning-rate 0.2 \
  --no-progress >/dev/null

test -s "$xgboost_tmp/scaling/compute_scaling.json"
test -s "$xgboost_tmp/scaling/compute_scaling.tsv"

python -m tcga_ml.candidate_comparison_cli \
  --evidence "$benchmark_tmp/out/benchmark.json" \
  --evidence "$feature_tmp/out/feature_budget.json" \
  --evidence "$xgboost_tmp/cv/xgboost_benchmark.json" \
  --outdir "$comparison_tmp/out" >/dev/null

test -s "$comparison_tmp/out/candidate_comparison.json"
test -s "$comparison_tmp/out/candidate_comparison.tsv"

python - "$focused_tmp" <<'PY'
import csv
import sys
from pathlib import Path
import numpy as np

root = Path(sys.argv[1])
rng = np.random.default_rng(11)
labels = ["LUAD", "LUSC", "KIRC", "KIRP"]
blocks = []
manifest = []
cache_index = 0
for class_index, label in enumerate(labels):
    block = rng.lognormal(0.4, 0.15, size=(8, 8)).astype("float32")
    block[:, class_index] *= 7
    blocks.append(block)
    for local_index in range(8):
        manifest.append(
            [
                cache_index,
                f"TCGA-{class_index:02d}-{local_index:04d}",
                label,
                "holdout" if local_index == 7 else "development",
            ]
        )
        cache_index += 1
np.save(root / "x.npy", np.vstack(blocks))
with (root / "split.tsv").open("w", newline="") as handle:
    writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
    writer.writerow(["cache_index", "participant_barcode", "cancer_type", "split"])
    writer.writerows(manifest)
with (root / "genes.tsv").open("w", newline="") as handle:
    writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
    writer.writerow(["gene_index", "source_gene_id", "symbol", "entrez_id"])
    for index in range(8):
        writer.writerow([index, f"PAIR{index}|{index}", f"PAIR{index}", index])
PY

python -m tcga_ml.focused_pairs_cli \
  --matrix "$focused_tmp/x.npy" \
  --split "$focused_tmp/split.tsv" \
  --genes "$focused_tmp/genes.tsv" \
  --outdir "$focused_tmp/out" \
  --model logistic_l2 \
  --gene-budget 4 \
  --cv-folds 2 \
  --n-jobs 1 \
  --no-progress >/dev/null

test -s "$focused_tmp/out/focused_pairs.json"
test -s "$focused_tmp/out/focused_pair_metrics.tsv"
test -s "$focused_tmp/out/focused_pair_confusion.tsv"
test -s "$focused_tmp/out/focused_pair_predictions.tsv"
test -s "$focused_tmp/out/focused_pair_errors.tsv"
test -s "$focused_tmp/out/focused_pair_genes.tsv"


python - "$final_tmp" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
(root / "selection.json").write_text(
    json.dumps(
        {
            "candidate_id": "ci-logistic-l2-four-genes",
            "primary_metric": "macro_f1",
            "selection_rationale": (
                "Synthetic CI candidate selected from development-only fixture evidence."
            ),
            "pipeline": {
                "family": "linear_gene_budget",
                "model": "logistic_l2",
                "gene_budget": 4,
                "negative_policy": "error",
                "scaler": "standard",
                "seed": 20260825,
            },
        }
    )
    + "\n",
    encoding="utf-8",
)
PY

python -m tcga_ml.final_evaluation_cli lock \
  --config "$final_tmp/selection.json" \
  --matrix "$benchmark_tmp/x.npy" \
  --split "$benchmark_tmp/split.tsv" \
  --genes "$benchmark_tmp/genes.tsv" \
  --evidence "$feature_tmp/out/feature_budget.json" \
  --output "$final_tmp/final.lock.json" >/dev/null

python -m tcga_ml.final_evaluation_cli verify \
  --lock "$final_tmp/final.lock.json" \
  --matrix "$benchmark_tmp/x.npy" \
  --split "$benchmark_tmp/split.tsv" \
  --genes "$benchmark_tmp/genes.tsv" \
  --evidence "$feature_tmp/out/feature_budget.json" >/dev/null

python -m tcga_ml.final_evaluation_cli evaluate \
  --lock "$final_tmp/final.lock.json" \
  --matrix "$benchmark_tmp/x.npy" \
  --split "$benchmark_tmp/split.tsv" \
  --genes "$benchmark_tmp/genes.tsv" \
  --evidence "$feature_tmp/out/feature_budget.json" \
  --outdir "$final_tmp/out" \
  --receipt "$final_tmp/final.receipt.json" >/dev/null

test -s "$final_tmp/final.receipt.json"
test -s "$final_tmp/out/final_metrics.json"
test -s "$final_tmp/out/final_predictions.tsv"
test -s "$final_tmp/out/final_confusion.tsv"
test -s "$final_tmp/out/final_pipeline.joblib"
test -s "$final_tmp/out/final_confusion.svg"
test -s "$final_tmp/out/final_per_class_f1.svg"

printf 'CI gate: PASS\n'
