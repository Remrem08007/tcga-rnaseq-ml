# TCGA RNA-seq ML — Study Roadmap

Status: **design locked before implementation**

This project will be an interpretable, leakage-resistant machine-learning study of TCGA PanCancer RNA-seq expression data. The goal is not only to classify tumor type, but to determine how much of the transcriptome is required for strong prediction, where models fail, how stable the selected genes are, and what compute/performance tradeoffs are involved.

> Research-use ML case study only. This repository will not claim clinical diagnostic validity.

## 1. Scientific questions

### Primary question
Can bulk tumor RNA expression distinguish selected TCGA cancer types on an untouched patient-level holdout set?

### Secondary questions
1. How few genes are needed before predictive performance meaningfully declines?
2. Which cancer pairs are most often confused, especially cancers from the same organ?
3. Which genes receive stable predictive importance across resamples/folds?
4. Do nonlinear models materially outperform regularized linear models?
5. What accuracy/runtime/memory tradeoffs are obtained from CPU parallelism and GPU acceleration?
6. How sensitive are conclusions to the feature-scaling strategy?

## 2. Initial cohort

Primary analysis: primary-tumor samples from ten well-represented cohorts:

- BRCA — Breast invasive carcinoma
- LUAD — Lung adenocarcinoma
- LUSC — Lung squamous cell carcinoma
- KIRC — Kidney renal clear cell carcinoma
- KIRP — Kidney renal papillary cell carcinoma
- LIHC — Liver hepatocellular carcinoma
- STAD — Stomach adenocarcinoma
- THCA — Thyroid carcinoma
- UCEC — Uterine corpus endometrial carcinoma
- COAD — Colon adenocarcinoma

Secondary focused comparisons:

- LUAD vs LUSC
- KIRC vs KIRP

The cohort builder will retain one eligible expression profile per participant and record every inclusion/exclusion decision.

## 3. Data source and provenance

Primary expression source:

`EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv`

from the NCI Genomic Data Commons TCGA PanCancer Atlas resources.

Additional public metadata will include cancer labels, TCGA barcodes, sample-quality annotations, and clinical/sample metadata needed for cohort construction and confounder audits.

Every source artifact will have a provenance record containing source URL, retrieval date, file size, and SHA-256 checksum. Large source matrices will not be committed to GitHub.

## 4. Normalization, transformation, and standardization

Normalization and standardization are deliberately separated.

### 4.1 Source-level normalization

The selected PanCancer expression matrix is already a processed/batch-corrected RNA expression resource. We will therefore **not** pretend that we are starting from raw read counts and will not apply DESeq2 size factors, CPM, TPM, TMM, or another count-level normalization to this matrix.

### 4.2 Data audit before transformation

Before modeling, an automated audit will report:

- minimum/maximum and quantiles of expression values;
- missing/non-finite values;
- fraction of zeros;
- per-gene variance;
- duplicated gene identifiers;
- sample/gene counts;
- whether values are compatible with the expected non-negative pre-log expression scale.

The analysis will fail loudly if the source representation does not match the configured transform rather than silently applying a second log transform.

### 4.3 Expression transform

For the intended PanCancer source representation, the working transform will be:

`x_log = log2(x + 1)`

only after the audit confirms that this is appropriate for the downloaded matrix.

No global quantile normalization will be added. The source is already normalized/batch-adjusted, and additional across-sample distribution forcing could remove real tissue/cancer biology.

### 4.4 Gene-wise standardization

For models that depend on feature scale, each gene will be standardized as:

`z = (x - training_mean) / training_sd`

Crucially, the mean and standard deviation are learned **only from the current training fold**. Validation and test samples are transformed with those stored training statistics.

This will be enforced using scikit-learn Pipelines so that scaling, feature selection, and model fitting remain inside cross-validation.

### 4.5 Model-specific scaling

- Logistic regression / elastic net: StandardScaler required.
- Linear SVM: StandardScaler required.
- PCA + linear classifier: scaling occurs before PCA and both are fit inside the fold.
- XGBoost: no z-score scaling required; it will use the audited/log-transformed expression values.
- Optional neural-network comparator: standardized input, with training-only scaler parameters.

### 4.6 Scaling sensitivity analysis

A predefined ablation will compare StandardScaler with RobustScaler for the primary linear model. This is a sensitivity check, not a post-hoc search for whichever preprocessing gives the best test result.

### 4.7 Leakage tests

Unit/integration tests will assert that:

- no participant occurs in both development and holdout sets;
- scalers are not fit on holdout samples;
- variance filtering and supervised feature selection are fit inside training folds;
- holdout labels are never used during model/hyperparameter selection;
- the final holdout manifest is deterministic and hashable.

## 5. Split and validation strategy

1. Build the eligible cohort and resolve duplicate participant samples deterministically.
2. Freeze a stratified patient-level 80/20 development/holdout split using a fixed seed.
3. Keep the 20% holdout untouched until the model-selection procedure is finalized.
4. Use stratified cross-validation on the 80% development set.
5. For the final report, use nested CV for model/hyperparameter comparison where computationally practical.
6. Refit the selected pipeline on all development samples.
7. Evaluate the locked model once on the final holdout.

Primary metric: **macro F1**.

Secondary metrics:

- balanced accuracy;
- overall accuracy;
- weighted F1;
- per-class precision, recall, and F1;
- confusion matrix;
- one-vs-rest ROC-AUC where appropriate;
- multiclass log loss for probabilistic models.

## 6. Models

### 6.1 Dummy classifier — sanity baseline

Purpose: prove that real models outperform trivial class-frequency behavior.

Strengths:
- essentially free to train;
- establishes a minimum baseline;
- catches broken evaluation pipelines.

Weaknesses:
- no biological or predictive value.

Why included: every benchmark needs a floor.

### 6.2 L2 multinomial logistic regression — stable linear baseline

Strengths:
- very strong baseline for high-dimensional omics;
- fast relative to nonlinear models;
- probabilistic outputs;
- coefficients are directly inspectable;
- L2 regularization handles correlated genes more stably than an unregularized model.

Weaknesses:
- linear decision boundaries;
- dense coefficient vectors are less useful for deriving small signatures;
- correlated genes can distribute importance across many coefficients.

Why included: tests whether broad transcriptomic separation is already mostly linear.

### 6.3 Elastic-net multinomial logistic regression — primary model

Strengths:
- designed for high-dimensional settings where features outnumber or approach sample counts;
- combines L1 sparsity with L2 stability;
- can produce compact gene signatures;
- interpretable class-specific coefficients;
- well suited to correlated biological features.

Weaknesses:
- still linear;
- selected genes can change when correlated alternatives carry similar information;
- SAGA optimization can be computationally demanding for large hyperparameter grids.

Why primary: best balance of prediction, interpretability, sparsity, and suitability for RNA expression.

### 6.4 Linear SVM — margin-based high-dimensional comparator

Strengths:
- often excellent when p >> n;
- focuses on decision margins rather than probability likelihood;
- efficient linear implementation;
- class-specific coefficients remain inspectable.

Weaknesses:
- native outputs are margins rather than calibrated probabilities;
- probability calibration adds another CV layer;
- coefficient magnitude can still be unstable under correlated features.

Why included: tests whether a max-margin objective improves on logistic loss.

### 6.5 PCA + logistic regression — compressed global-expression baseline

Strengths:
- greatly reduces dimensionality;
- efficient training after projection;
- tests whether broad global expression structure is sufficient;
- useful for visualization and runtime comparisons.

Weaknesses:
- principal components optimize variance, not class discrimination;
- components mix many genes and reduce biological interpretability;
- low-variance but predictive genes can be lost.

Why included: provides a principled dimension-reduction comparison rather than assuming gene-level selection is always better.

### 6.6 XGBoost — nonlinear comparator and primary GPU model

Strengths:
- captures nonlinear effects and gene-gene interactions;
- strong tabular-data performance;
- current XGBoost supports CUDA histogram training;
- supports feature importance/SHAP-style interpretation;
- QuantileDMatrix/external-memory paths can reduce memory pressure.

Weaknesses:
- 20k features with only thousands of patients creates substantial overfitting risk;
- less transparent than sparse linear models;
- hyperparameter tuning is expensive;
- feature importance can be unstable/correlated.

Why included: directly tests whether nonlinear structure adds meaningful held-out performance and gives us a defensible GPU workload.

### 6.7 Optional PyTorch MLP — GPU/deep-learning ablation

This is **not** a required primary model. It is added only after the classical benchmark is stable.

Strengths:
- native GPU acceleration;
- nonlinear interactions;
- connects naturally to prior PyTorch training/checkpointing experience;
- can use early stopping, dropout, mixed precision, and mini-batching.

Weaknesses:
- high overfitting risk on this sample size;
- harder to interpret;
- many additional design choices;
- may add complexity without improving generalization.

Why optional: it answers whether a modest neural network adds anything beyond strong classical models rather than adding “deep learning” for branding.

## 7. Feature-selection experiments

All learned feature selection occurs inside the training fold.

Planned gene-budget experiment:

- 20 genes
- 50 genes
- 100 genes
- 200 genes
- 500 genes
- 1,000 genes
- 5,000 genes
- all eligible genes

Candidate selector for controlled gene-budget comparisons: training-fold `SelectKBest(f_classif)` followed by scaling/model fitting. Elastic-net intrinsic sparsity will be analyzed separately.

Outputs:

- macro F1 vs gene count;
- runtime vs gene count;
- peak memory vs gene count;
- feature-selection stability across folds;
- class-specific influential genes.

Predictive importance will never be presented as causal biology.

## 8. Interpretation and error analysis

Primary outputs:

- normalized confusion matrix;
- per-class metrics;
- coefficient heatmaps for linear models;
- stable-gene selection frequencies;
- PCA projections;
- learning curves;
- performance vs gene-count curves;
- hardest cancer-pair analysis;
- optional SHAP analysis for XGBoost;
- misclassified sample audit.

We will specifically examine LUAD/LUSC and KIRC/KIRP because they test within-organ/histologic discrimination rather than easy tissue-of-origin differences.

## 9. Confounder and robustness audits

Where metadata allow, inspect associations between cancer label and:

- tissue source site;
- known sample-quality annotations;
- sequencing/batch variables;
- age/sex distributions;
- tumor purity or related available estimates.

The project will distinguish a predictive signal from evidence of causal tumor biology.

## 10. Compute and performance architecture

Efficiency is a design requirement, but optimizations must be justified by profiling.

### 10.1 Data representation

The downloaded TSV will be converted once into an analysis cache:

- expression matrix stored as `float32`;
- sample × gene orientation for ML;
- memory-mappable `.npy` array for fast repeated reads;
- separate compact sample/gene metadata tables;
- source checksum and transform metadata stored beside the cache.

The converter should stream/chunk the original gene × sample TSV into the transposed memory-mapped array rather than materializing multiple float64 copies.

### 10.2 CPU parallelization

Use parallelism at the level that provides independent work:

- cross-validation folds;
- model families;
- randomized hyperparameter candidates;
- gene-budget experiments.

Implementation options:

- scikit-learn/joblib `n_jobs` on a single node;
- SLURM arrays for coarse independent experiment shards on HPC;
- limit BLAS/OpenMP threads inside each worker to avoid oversubscription.

Large input arrays should be memory-mapped/shared rather than copied once per worker.

### 10.3 GPU acceleration

Primary GPU target: XGBoost with `device="cuda"` and histogram tree construction.

Optional GPU targets:

- PyTorch MLP with mixed precision when supported;
- GPU PCA only if profiling shows PCA is a meaningful bottleneck.

We will not force GPU use for algorithms whose CPU implementations are already faster/simpler at this dataset size.

### 10.4 Native-code/JIT acceleration

We will **not hand-write assembly or blindly translate Python to assembly**.

Most expensive numerical operations in NumPy/scikit-learn already execute in compiled C/C++/Fortran/BLAS code, and XGBoost is native C++ with CUDA support.

Optimization order:

1. vectorize Python loops with NumPy;
2. use efficient dtypes/layout (`float32`, contiguous arrays, memmaps);
3. parallelize independent work;
4. profile;
5. use Numba `@njit`/`parallel=True` for proven custom Python hotspots, compiling them through LLVM to native machine code;
6. only consider Cython/C++/Rust extensions if profiling shows a persistent hotspot that cannot be removed with existing libraries.

This gives the performance benefits associated with native code without sacrificing maintainability for cosmetic low-level code.

### 10.5 Compute benchmark

The final project will include a reproducible compute benchmark:

- CPU worker count: 1 / 2 / 4 / 8 / 16 where available;
- wall-clock time;
- peak RSS;
- model macro F1;
- GPU XGBoost wall-clock time when CUDA is available;
- optional CPU-vs-GPU speedup plot.

Performance benchmarking will be reported separately from biological predictive performance.

## 11. Lessons carried forward from previous ML work

Patterns worth retaining from earlier projects:

- grouped data splitting instead of blindly randomizing related observations;
- explicit train/validation/test artifacts;
- reproducible seeds;
- saved model/checkpoint/history artifacts;
- configurable experiments rather than one-off notebooks;
- custom metric validation and threshold/metric summaries;
- HDF5/structured binary data experience;
- GPU-aware PyTorch/Flux workflows;
- comparing raw/simple representations with learned/embedded representations;
- strong visual diagnostics of training and predictions.

One deliberate improvement over older experimental code is that **all learned preprocessing parameters will be fit after splitting and inside CV**. In particular, z-score statistics will never be computed using the future holdout set.

## 12. Proposed repository architecture

```text
tcga-rnaseq-ml/
├── README.md
├── ROADMAP.md
├── LICENSE
├── pyproject.toml
├── config/
│   ├── cohort.yaml
│   └── models.yaml
├── data/
│   └── README.md
├── src/
│   └── tcga_ml/
│       ├── barcodes.py
│       ├── provenance.py
│       ├── ingest.py
│       ├── cohort.py
│       ├── normalization.py
│       ├── splitting.py
│       ├── features.py
│       ├── models.py
│       ├── tuning.py
│       ├── evaluation.py
│       ├── interpretation.py
│       └── compute.py
├── scripts/
│   ├── download_data.py
│   ├── build_cache.py
│   ├── build_cohort.py
│   ├── run_benchmark.py
│   └── make_report.py
├── slurm/
│   ├── benchmark.sbatch
│   └── benchmark_array.sbatch
├── tests/
├── notebooks/
│   ├── 01_cohort_qc.ipynb
│   ├── 02_expression_eda.ipynb
│   ├── 03_model_comparison.ipynb
│   └── 04_interpretation.ipynb
└── results/
    ├── metrics/
    ├── predictions/
    ├── selected_genes/
    ├── compute/
    └── figures/
```

Notebooks will be presentation/exploration layers; core analysis logic belongs in tested modules under `src/`.

## 13. Milestones

### M0 — Design and provenance
- lock scientific question and cohorts;
- define source files and checksums;
- freeze normalization/standardization policy;
- define evaluation contract and seed.

### M1 — Ingest and cohort builder
- download/provenance tooling;
- TCGA barcode parser;
- primary-tumor and quality filters;
- one participant/sample policy;
- chunked `float32` memory-mapped expression cache;
- cohort manifest and tests.

### M2 — Leakage-safe preprocessing
- data audit;
- `log2(x+1)` transform guard;
- variance/missingness filters;
- StandardScaler/RobustScaler pipelines;
- deterministic dev/holdout split;
- leakage tests.

### M3 — Classical baseline benchmark
- dummy;
- L2 logistic regression;
- elastic-net logistic regression;
- linear SVM;
- PCA + logistic regression;
- CV metrics/runtime summaries.

### M4 — Feature-budget and stability study
- 20 → all-gene feature budgets;
- selection stability;
- coefficient interpretation;
- learning/performance curves.

### M5 — Nonlinear/GPU benchmark
- XGBoost CPU and CUDA paths;
- optional PyTorch MLP if justified;
- CPU/GPU runtime and memory benchmark.

### M6 — Focused cancer-pair studies
- LUAD vs LUSC;
- KIRC vs KIRP;
- error analysis and interpretation.

### M7 — Final locked evaluation
- select final pipeline using development data only;
- refit on full development set;
- evaluate untouched holdout once;
- save predictions and metrics;
- generate final figures.

### M8 — Portfolio release
- complete README/results summary;
- methods, leakage, normalization, compute, and limitations documentation;
- CI and tests;
- reproducible small synthetic fixture;
- public release tag.

## 14. Definition of success

A successful project is not defined by reaching an arbitrary accuracy threshold. It should demonstrate:

1. reproducible patient-level cohort construction;
2. correct normalization/standardization semantics;
3. no preprocessing or feature-selection leakage;
4. transparent comparison of strong classical and nonlinear models;
5. interpretable gene-level findings with stability analysis;
6. a useful reduced-gene performance study;
7. scientifically interesting error analysis;
8. measured CPU/GPU performance rather than unsubstantiated acceleration claims;
9. tests, provenance, and reproducible outputs;
10. an explicit statement that TCGA retrospective classification is not clinical diagnostic validation.
