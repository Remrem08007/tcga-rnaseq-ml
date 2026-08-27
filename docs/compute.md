# Compute and acceleration policy

This project treats compute optimization as an empirical engineering question, not as a reason to use more complex hardware or lower-level code by default. Predictive performance and compute performance are reported separately.

## Why XGBoost is the nonlinear comparator

The primary interpretable model remains elastic-net multinomial logistic regression. XGBoost is included because it asks a different question: whether nonlinear effects and gene-gene interactions add useful predictive signal beyond strong linear models.

Strengths:

- nonlinear decision rules and interaction modeling;
- strong tabular-data performance;
- native C++ implementation;
- CUDA histogram training in current XGBoost;
- fold-level feature importance that can be mapped back to genes.

Weaknesses:

- substantially greater overfitting risk with thousands of genes and only thousands of patients;
- more tuning choices than a linear model;
- less direct biological interpretation;
- correlated genes can make tree importance unstable;
- GPU training is not automatically faster for every dataset size or gene budget.

For that reason, XGBoost is a comparator rather than the default model. A GPU speedup is useful only if predictive performance remains comparable and the benchmark is measured fairly.

## Preprocessing contract

The XGBoost pipeline is fit independently inside every development-set CV training fold:

```text
log2p1 -> median imputation -> zero-variance filtering -> SelectKBest -> XGBoost
```

The frozen holdout is not used by these experiments.

Unlike the linear models, XGBoost does not use `StandardScaler`. Histogram trees depend on feature ordering and split thresholds, so z-standardization does not provide the optimization benefit it gives linear models. Removing it also avoids an unnecessary full-matrix transform.

The optional `SelectKBest(f_classif)` gene budget remains inside each fold. It is a controlled dimensionality experiment, not a claim that the selected genes are causal.

## CPU parallelization

XGBoost exposes native CPU threading through `n_jobs`. Cross-validation can also be parallelized across folds, but using both levels without limits creates oversubscription.

The benchmark therefore enforces:

```text
fold_jobs * threads <= available CPU allocation
```

On SLURM, the available allocation is read from `SLURM_CPUS_PER_TASK` when present. For compute-scaling experiments, `fold_jobs=1` is used so 1/2/4/8/16-thread measurements compare one level of parallelism at a time.

For the classical scikit-learn models, fold-level joblib parallelism remains appropriate because the underlying estimators are mostly single-worker in our configuration and BLAS/OpenMP threads are limited inside workers.

## CUDA policy

Current XGBoost uses histogram tree construction with `device="cuda"` for GPU training. Merely having a CUDA-enabled wheel is not sufficient evidence that a run used the GPU: when no compatible device is visible, XGBoost can warn and fall back to CPU.

`python -m tcga_ml.xgboost_cli probe` therefore performs a small real training run and inspects the fitted booster configuration. An explicit `--device cuda` benchmark fails if the booster did not remain on CUDA. This prevents a CPU fallback from being reported as GPU acceleration.

GPU CV uses `fold_jobs=1` so multiple folds do not compete for one allocated GPU. CPU-side XGBoost threads are still available for data preparation and host work.

## What is measured

The reproducible scaling benchmark records:

- wall-clock time;
- speedup versus the first CPU configuration supplied;
- macro F1;
- balanced accuracy;
- controller-process peak host RSS where supported;
- requested/resolved device and CPU-thread count.

Host RSS is not labeled as GPU memory. A true GPU peak-memory result requires a device-level sampler (for example NVML/nvidia-smi sampling) synchronized with the training process; until that is implemented, the project does not invent or infer a GPU peak-memory number.

## Native code and assembly

Hand-written assembly is not justified here. NumPy/scikit-learn already execute their expensive numerical kernels in compiled C/C++/Fortran/BLAS code, and XGBoost itself is native C++ with CUDA kernels.

Optimization order is therefore:

1. memory-map the `float32` expression cache;
2. avoid unnecessary copies/transforms;
3. vectorize Python work;
4. parallelize independent work without oversubscription;
5. profile;
6. use Numba/Cython/C++ only for a demonstrated custom Python hotspot.

This gives native-machine-code performance where it matters without sacrificing maintainability for cosmetic low-level code.

## Optional PyTorch MLP

A small PyTorch MLP remains an optional later ablation, not a required milestone. It would provide a second nonlinear GPU workload and support dropout, early stopping, mini-batching, and mixed precision, but it also adds substantial overfitting and tuning risk. It should be added only after the XGBoost/classical comparison is stable enough to show whether another nonlinear model answers a meaningful question.

## SLURM templates

Two generic starting points are provided:

- `slurm/xgboost_cpu_scaling.sbatch` for CPU scaling;
- `slurm/xgboost_gpu.sbatch` for an explicit CUDA run.

They intentionally contain no private account name, allocation, project path, or cluster-specific partition. Set `REPO_ROOT`, `MATRIX`, `SPLIT`, `GENES`, and `OUTDIR` in the job environment when the defaults do not match your staged analysis directory.
