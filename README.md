# TCGA RNA-seq ML

An interpretable, leakage-resistant machine-learning case study using the TCGA PanCancer Atlas batch-corrected RNA-seq expression matrix.

The project is being built in stages. **No clinical diagnostic claim is made**; TCGA samples are retrospective research specimens with known cancer labels.

## Current implementation

Implemented and tested:

- official GDC source registry + download provenance;
- TCGA barcode parsing;
- primary-solid-tumor / quality / platform / cancer-type cohort filtering;
- deterministic one-expression-profile-per-participant selection;
- chunked `float32` sample × gene memory-mappable cache;
- expression-value audit before normalization;
- explicit `log2(x + 1)` transformer with no silent negative clipping;
- train-only imputation, zero-variance filtering, StandardScaler / RobustScaler pipelines;
- frozen stratified 80/20 participant-level development/holdout manifest with SHA-256;
- development-only CV benchmark for dummy, L2 logistic, elastic net, linear SVM, and PCA + logistic models;
- CV parallelism with inner numerical threads limited to avoid CPU oversubscription;
- leakage-safe gene-budget experiments with fold-wise ANOVA selection, selection stability, and class-specific linear coefficients;
- nonlinear XGBoost benchmark with inverse-frequency fold-local sample weights;
- explicit CPU/CUDA device handling that refuses silent GPU-to-CPU fallback;
- CPU thread-scaling and optional GPU timing benchmark with predictive metrics kept alongside timing results.

See [`ROADMAP.md`](ROADMAP.md) for the locked study design and remaining milestones and [`docs/compute.md`](docs/compute.md) for the acceleration policy.

## Reproducible data setup

Create an environment and install the development dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,xgboost]'
```

For a runtime installation that only needs the nonlinear benchmark, install the optional XGBoost extra:

```bash
python -m pip install -e '.[xgboost]'
```

List the registered open-access GDC files:

```bash
python -m tcga_ml.download --list
```

Download the two M1 inputs:

```bash
python -m tcga_ml.download --source sample_quality
python -m tcga_ml.download --source expression
```

Build the locked ten-cancer primary-tumor cohort:

```bash
python -m tcga_ml.cohort_cli \
  --expression data/raw/EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv \
  --quality data/raw/merged_sample_quality_annotations.tsv \
  --outdir data/processed
```

Build the `float32` memory-mappable cache:

```bash
python -m tcga_ml.cache_cli \
  --expression data/raw/EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv \
  --cohort data/processed/cohort.tsv \
  --outdir data/cache
```

Audit the source representation **before choosing the negative-value policy**:

```bash
python -m tcga_ml.audit_cli \
  --matrix data/cache/expression.float32.npy \
  --output data/cache/expression_audit.json
```

Freeze the participant-level holdout:

```bash
python -m tcga_ml.split_cli \
  --samples data/cache/samples.tsv \
  --outdir data/processed \
  --holdout-fraction 0.20 \
  --seed 20260825
```

Only after reviewing the expression audit, run development CV. If small negative values from batch adjustment are confirmed and clipping them to zero is accepted for this analysis, opt in explicitly with `--negative-policy clip`:

```bash
python -m tcga_ml.benchmark_cli \
  --matrix data/cache/expression.float32.npy \
  --split data/processed/split_manifest.tsv \
  --outdir results/benchmarks/classical \
  --negative-policy clip \
  --n-jobs 0
```

The benchmark command **does not evaluate the frozen holdout**.

## Gene-budget and feature-stability study

Run the M4 gene-budget/stability study with the primary elastic-net model:

```bash
python -m tcga_ml.feature_budget_cli \
  --matrix data/cache/expression.float32.npy \
  --split data/processed/split_manifest.tsv \
  --genes data/cache/genes.tsv \
  --outdir results/feature_budget/elastic_net \
  --model elastic_net \
  --negative-policy clip \
  --n-jobs 0
```

By default this evaluates the locked `20, 50, 100, 200, 500, 1000, 5000, all` gene budgets using development-set CV only. It writes:

- `feature_budget.json` — performance, fit/score time, wall time, and controller-process memory observations by budget;
- `feature_stability.tsv` — how often each gene is selected across folds plus mean absolute linear-model coefficient;
- `coefficients.tsv` — fold- and class-specific coefficients mapped back to the original TCGA gene index/symbol.

Feature selection remains inside each training fold: `log2p1 → imputation → variance filtering → SelectKBest → scaling → model`. The frozen holdout is still untouched.

## Nonlinear XGBoost benchmark

The XGBoost path deliberately does **not** z-standardize expression values: histogram trees are invariant to monotonic feature rescaling. The fold-local pipeline is:

```text
log2p1 → median imputation → variance filtering → SelectKBest → XGBoost
```

Run the CPU comparator:

```bash
python -m tcga_ml.xgboost_cli cv \
  --matrix data/cache/expression.float32.npy \
  --split data/processed/split_manifest.tsv \
  --genes data/cache/genes.tsv \
  --outdir results/benchmarks/xgboost-cpu \
  --device cpu \
  --threads 8 \
  --fold-jobs 1 \
  --gene-budget 1000 \
  --negative-policy clip
```

Before a GPU run, probe the actual runtime:

```bash
python -m tcga_ml.xgboost_cli probe
```

Then, on a CUDA-capable allocated node:

```bash
python -m tcga_ml.xgboost_cli cv \
  --matrix data/cache/expression.float32.npy \
  --split data/processed/split_manifest.tsv \
  --genes data/cache/genes.tsv \
  --outdir results/benchmarks/xgboost-gpu \
  --device cuda \
  --threads 4 \
  --fold-jobs 1 \
  --gene-budget 1000 \
  --negative-policy clip
```

The CUDA probe performs real XGBoost training and inspects the fitted booster device. If XGBoost warns and silently falls back to CPU, an explicit `--device cuda` run fails instead of being mislabeled as GPU acceleration.

XGBoost writes `xgboost_benchmark.json` plus `xgboost_feature_importance.tsv`, where fold-wise feature importance is mapped back to original TCGA gene identifiers. This remains **development-set CV only**.

## CPU/GPU compute benchmark

Measure CPU thread scaling without concurrent folds competing for the same allocation:

```bash
python -m tcga_ml.xgboost_cli scale \
  --matrix data/cache/expression.float32.npy \
  --split data/processed/split_manifest.tsv \
  --outdir results/compute/xgboost \
  --cpu-threads 1 \
  --cpu-threads 2 \
  --cpu-threads 4 \
  --cpu-threads 8 \
  --cpu-threads 16 \
  --gene-budget 1000 \
  --negative-policy clip
```

Add `--include-gpu` on a GPU node to append a CUDA measurement, or `--require-gpu` when an unavailable CUDA path should fail the job. The benchmark records wall time, speedup relative to the first CPU run, macro F1, balanced accuracy, and host RSS. It does **not** claim a GPU peak-memory value without a proper device-level sampler.

Generic SLURM starting points are provided in `slurm/xgboost_cpu_scaling.sbatch` and `slurm/xgboost_gpu.sbatch`; cluster-specific accounts/partitions remain intentionally absent from the public repository.

## Green-commit rule

Every code commit must pass the same repository gate locally before publication:

```bash
scripts/ci_local.sh
```

GitHub Actions invokes the same script on Python 3.11 and 3.12, so local and hosted CI do not maintain separate test logic.

## Data policy

Large TCGA source and derived matrices are ignored by Git and are never committed. Source downloads receive provenance sidecars with GDC URL, retrieval time, size, and SHA-256.

## License

A license will be added before the first public release tag.
