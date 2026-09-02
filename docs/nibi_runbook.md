# Nibi analysis runbook

This runbook continues from a built cohort/cache/frozen split and a completed
elastic-net feature-budget study. The commands through candidate comparison use
development participants only. They do not load or score the frozen holdout.

## 1. Update the code and environment

Run these commands on a Nibi login node:

```bash
module load StdEnv/2023
module load python/3.11

REPO="/project/6055534/remrem08/tcga-rnaseq-ml"
cd "$REPO"

git switch main
git pull --ff-only
source "$REPO/.venv/bin/activate"
python -m pip install -e '.[dev,xgboost]'

mkdir -p logs results/benchmarks results/compute results/focused_pairs results/final
```

The editable install refreshes the existing virtual environment with the latest
project code and ensures the optional XGBoost dependency is present. It does not
touch the TCGA data or result files.

## 2. Confirm the completed feature-budget artifact

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("results/feature_budget/elastic_net/feature_budget.json")
data = json.loads(path.read_text())
observed = [row["gene_budget"] for row in data["budgets"]]
expected = [20, 50, 100, 200, 500, 1000, 5000, "all"]
assert data["evaluation_scope"] == "development_cross_validation_only"
assert data["holdout_used"] is False
assert observed == expected, observed
print("feature budget: COMPLETE", observed)
PY
```

## 3. Submit the remaining development analyses

The five jobs are independent and may run concurrently:

```bash
CLASSICAL_JOB=$(sbatch --parsable --account=def-rejlap slurm/classical_benchmark.sbatch)
XGB_CV_JOB=$(sbatch --parsable --account=def-rejlap slurm/xgboost_cpu_cv.sbatch)
XGB_SCALE_JOB=$(sbatch --parsable --account=def-rejlap slurm/xgboost_cpu_scaling.sbatch)
XGB_GPU_JOB=$(sbatch --parsable --account=def-rejlap slurm/xgboost_gpu.sbatch)
PAIRS_JOB=$(sbatch --parsable --account=def-rejlap slurm/focused_pairs.sbatch)

printf 'classical=%s\nxgboost_cv=%s\nxgboost_scaling=%s\nxgboost_gpu=%s\nfocused_pairs=%s\n' \
  "$CLASSICAL_JOB" "$XGB_CV_JOB" "$XGB_SCALE_JOB" "$XGB_GPU_JOB" "$PAIRS_JOB"
```

What each job answers:

| Job | Scientific/engineering question | Main output |
| --- | --- | --- |
| Classical benchmark | How do dummy, L2 logistic, elastic net, linear SVM, and PCA-logistic compare using all input genes? | `results/benchmarks/classical/benchmark.json` |
| XGBoost CPU CV | Does a nonlinear tree model add development-CV predictive performance at a 1,000-gene budget? | `results/benchmarks/xgboost-cpu/xgboost_benchmark.json` |
| XGBoost scaling | How much runtime changes at 1, 2, 4, 8, and 16 CPU threads, while predictive metrics are held alongside timing? | `results/compute/xgboost-cpu/compute_scaling.json` |
| XGBoost GPU CV | Can the same nonlinear comparator train on a verified CUDA device, and how does its runtime compare? | `results/benchmarks/xgboost-gpu/xgboost_benchmark.json` |
| Focused pairs | Where do LUAD/LUSC and KIRC/KIRP fail, and which selected-gene directions are stable across folds? | `results/focused_pairs/elastic_net/focused_pairs.json` |

The XGBoost CV intentionally uses one fold at a time and all eight XGBoost
threads inside that fold. The scaling study also uses one fold at a time so the
thread-count comparison is not confounded by folds competing for the same CPUs.
The classical and focused-pair jobs instead parallelize independent folds and
limit numerical libraries inside each worker to one thread.

The Nibi GPU template requests one `h100_1g.10gb` MIG instance. It performs a
real CUDA training probe before the benchmark and fails if XGBoost silently
falls back to CPU. The GPU result is useful compute evidence but is not required
for the holdout-blind candidate table, whose XGBoost entry is the reproducible
CPU comparator.

## 4. Monitor jobs and progress

```bash
squeue -j "$CLASSICAL_JOB,$XGB_CV_JOB,$XGB_SCALE_JOB,$XGB_GPU_JOB,$PAIRS_JOB" \
  -o "%.18i %.10T %.10M %.10l %.6D %R"
```

Each log receives a scheduler-safe progress line after a fold finishes and a
heartbeat every minute while a fold is still running. Follow any one log with:

```bash
tail -F "logs/tcga-xgb-cv-${XGB_CV_JOB}.out"
```

Stopping `tail` with `Ctrl-C` does not stop the job. Check final scheduler
accounting with:

```bash
sacct -j "$CLASSICAL_JOB,$XGB_CV_JOB,$XGB_SCALE_JOB,$XGB_GPU_JOB,$PAIRS_JOB" \
  --format=JobID,JobName%24,State,ExitCode,Elapsed,MaxRSS
```

`COMPLETED` with exit code `0:0` is the scheduler-level check. Validate the
application outputs as well:

```bash
python - <<'PY'
import json
from pathlib import Path

checks = {
    "classical": Path("results/benchmarks/classical/benchmark.json"),
    "xgboost_cv": Path("results/benchmarks/xgboost-cpu/xgboost_benchmark.json"),
    "xgboost_scaling": Path("results/compute/xgboost-cpu/compute_scaling.json"),
    "xgboost_gpu": Path("results/benchmarks/xgboost-gpu/xgboost_benchmark.json"),
    "focused_pairs": Path("results/focused_pairs/elastic_net/focused_pairs.json"),
}
for name, path in checks.items():
    if not path.is_file():
        print(f"{name}: INCOMPLETE ({path} missing)")
        continue
    payload = json.loads(path.read_text())
    safe = (
        payload.get("evaluation_scope") == "development_cross_validation_only"
        and payload.get("holdout_used") is False
    )
    print(f"{name}: {'COMPLETE' if safe else 'CHECK METADATA'} ({path})")
PY
```

## 5. Build the holdout-blind candidate table

Run this only after the classical and XGBoost CV JSON files are complete:

```bash
python -m tcga_ml.candidate_comparison_cli \
  --evidence results/benchmarks/classical/benchmark.json \
  --evidence results/feature_budget/elastic_net/feature_budget.json \
  --evidence results/benchmarks/xgboost-cpu/xgboost_benchmark.json \
  --outdir results/final/candidate-comparison

column -s $'\t' -t \
  results/final/candidate-comparison/candidate_comparison.tsv | less -S
```

This command only combines already-computed development metrics. It rejects an
artifact unless it explicitly says `holdout_used: false`, ranks candidates by
mean development-CV macro F1, and deliberately does not select a winner.

The output directory is created with non-overwriting semantics. If it already
exists, inspect the existing result rather than deleting it automatically.

## 6. Stop before the one-time holdout boundary

Review these files together before writing `config/final_pipeline.json`:

```bash
column -s $'\t' -t \
  results/final/candidate-comparison/candidate_comparison.tsv | less -S

column -s $'\t' -t \
  results/focused_pairs/elastic_net/focused_pair_metrics.tsv | less -S

column -s $'\t' -t \
  results/compute/xgboost-cpu/compute_scaling.tsv | less -S
```

Do not run `final_evaluation_cli evaluate` yet. Candidate selection, rationale,
lock creation, and lock verification are the next reviewed stage. The real
`evaluate` command creates an irreversible receipt before reading holdout rows.

## Progress behavior

Progress reporting is enabled by default for the classical, feature-budget,
XGBoost, and focused-pair CLIs. Interactive terminals refresh one line; Slurm
logs receive durable line-oriented updates. `--no-progress` disables both fold
updates and the one-minute heartbeat when quiet output is required.
