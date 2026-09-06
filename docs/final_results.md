# Final study results

## Scope

This repository implements a retrospective, research-use machine-learning study of
the TCGA PanCancer Atlas batch-corrected RNA-seq matrix. It is not a clinical
diagnostic model and has not been validated prospectively, externally, or for use
on patient-care samples.

The analysis includes 4,599 participants from ten TCGA cancer cohorts and 20,531
expression features. A deterministic, stratified participant-level split reserved
3,679 participants for development and 920 participants for a frozen final
holdout.

## Leakage-control design

The following rules were fixed before the final evaluation:

- participants, rather than expression columns alone, define the split boundary;
- imputation, variance filtering, supervised gene selection, and scaling are fit
  only on the current training data;
- model and gene-budget comparisons use five-fold stratified cross-validation on
  the development set;
- focused-pair results use exactly one out-of-fold prediction per development
  participant;
- the holdout is excluded from preprocessing, feature selection, fitting, model
  comparison, and error analysis until the one-time final evaluation;
- the selected pipeline, primary metric, evidence files, split, matrix, and gene
  manifest are bound in a SHA-256 lock before holdout access.

The audited source contained a small fraction of negative batch-adjusted values.
The locked analysis used the explicit `clip` policy before `log2(x + 1)`.
Non-finite values were imputed with training-fold medians.

## Development-set model selection

The primary metric was macro F1. The development-only comparison ranked the
all-gene elastic net first, but selected the 5,000-gene elastic net because it
retained essentially the same predictive performance with a substantially lower
runtime.

| Development candidate | Genes | Macro F1, mean ± SD | Balanced accuracy | Wall time |
|---|---:|---:|---:|---:|
| Elastic net | all | 0.983568 ± 0.003819 | 0.983847 | 14,644.8 s |
| **Elastic net (selected)** | **5,000** | **0.982697 ± 0.003951** | **0.983366** | **3,243.4 s** |
| XGBoost, CPU | 1,000 | 0.980655 ± 0.004651 | 0.981024 | 254.4 s |
| L2 logistic regression | all | 0.979026 ± 0.005765 | 0.980397 | 12.0 s |
| Linear SVM | all | 0.975881 ± 0.005101 | 0.978553 | 89.4 s |

The all-gene elastic net improved mean macro F1 by only 0.000871 (0.087
percentage points), won three of five folds, and required approximately 4.5 times
the wall time. The selected configuration was therefore the prespecified
performance/compute tradeoff rather than the numerically highest mean alone.

### Selected pipeline

- family: `linear_gene_budget`
- model: multinomial elastic-net logistic regression
- gene budget: 5,000 of 20,531 input genes
- gene selector: training-only ANOVA `SelectKBest(f_classif)`
- scaler: training-only standard scaling
- negative-value policy: explicit clipping
- class weighting: balanced
- solver: SAGA
- `C=1`, `l1_ratio=0.5`, `max_iter=5000`
- seed: `20260825`

The final classifier contains one 5,000-element coefficient vector and one
intercept per cancer class: 50,000 coefficients and 10 intercepts.

## Feature-selection stability

Each cross-validation fold selected 5,000 genes using only its training
participants.

| Folds selecting a gene | Number of genes |
|---:|---:|
| 5 of 5 | 4,734 |
| 4 of 5 | 142 |
| 3 of 5 | 119 |
| 2 of 5 | 118 |
| 1 of 5 | 169 |

Across folds, 5,282 unique genes were selected at least once. The 4,734-gene
intersection represents 94.68% of each fold's 5,000-gene set and 89.63% of the
union. This supports a stable broad predictive signature while also showing that
correlated alternatives can move near the selection boundary.

## One-time frozen-holdout result

The locked pipeline was refit on all 3,679 development participants and evaluated
once on the 920-participant holdout.

| Metric | Final holdout |
|---|---:|
| **Macro F1 (primary)** | **0.980828** |
| Balanced accuracy | 0.980087 |
| Accuracy | 0.981522 |
| Weighted F1 | 0.981544 |
| Multiclass log loss | 0.066979 |
| Macro one-vs-rest ROC AUC | 0.999516 |
| Errors | 17 of 920 |

The holdout macro F1 was 0.001869 (0.187 percentage points) below the selected
candidate's development-CV mean and remained within its fold-to-fold variability.

### Per-class performance

| Cancer | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| BRCA | 217 | 1.0000 | 1.0000 | 1.0000 |
| COAD | 56 | 1.0000 | 1.0000 | 1.0000 |
| KIRC | 103 | 0.9619 | 0.9806 | 0.9712 |
| KIRP | 57 | 0.9643 | 0.9474 | 0.9558 |
| LIHC | 74 | 1.0000 | 0.9865 | 0.9932 |
| LUAD | 102 | 0.9423 | 0.9608 | 0.9515 |
| LUSC | 97 | 0.9479 | 0.9381 | 0.9430 |
| STAD | 80 | 1.0000 | 0.9875 | 0.9937 |
| THCA | 100 | 1.0000 | 1.0000 | 1.0000 |
| UCEC | 34 | 1.0000 | 1.0000 | 1.0000 |

### Error pattern

| True class | Predicted class | Count |
|---|---|---:|
| KIRC | KIRP | 2 |
| KIRP | KIRC | 3 |
| LIHC | LUSC | 1 |
| LUAD | KIRC | 1 |
| LUAD | LUSC | 3 |
| LUSC | LUAD | 6 |
| STAD | LUSC | 1 |

Fourteen of the 17 errors (82.4%) fell within the two similar-cancer pairs
prespecified for development analysis: LUAD/LUSC and KIRC/KIRP.

The development-only focused studies had macro F1 of 0.9523 for LUAD versus LUSC
(797 participants, 38 out-of-fold errors) and 0.9543 for KIRC versus KIRP (640
participants, 27 out-of-fold errors). The final error pattern is therefore
consistent with the difficulty identified before the holdout was opened.

## Compute findings

The XGBoost comparator reached development macro F1 0.9807 using 1,000 genes. In
the measured CPU scaling experiment, increasing the XGBoost thread count from 1
to 16 produced little additional speedup: 346.6 s at one thread versus 323.3 s at
16 threads, with the best measured CPU time of 317.5 s at two threads.

On the tested H100 MIG allocation, the CUDA-verified XGBoost run completed in
52.1 s versus 254.4 s for the corresponding CPU benchmark, approximately 4.9
times faster. The implementation performs a real CUDA training probe and checks
the fitted booster device so silent fallback to CPU cannot be reported as GPU
acceleration. This is a compute result, not evidence that XGBoost predicted
better than the selected linear model.

## Numerical diagnostics

The saved final SAGA model reached its configured `max_iter=5000` and emitted a
convergence warning. The reported classification metrics belong to that exact
saved model, but full numerical convergence was not certified. Coefficient
magnitudes and probability calibration should therefore be interpreted
conservatively. The official holdout result was not rerun with altered training
settings after observing the holdout.

Scikit-learn also warned that probability rows did not sum exactly to one. A
post-run audit of the saved predictions found a maximum row-sum deviation of
2.30e-7, with no row deviating by more than 1e-6. This is floating-point
roundoff, not a material probability-normalization failure.

## Reproducibility record

- evaluation code commit: `92d1e50613175f21c16f88585c61c72867684ae6`
- split seed: `20260825`
- final lock SHA-256:
  `38875d0d05fadfe2dc86ea5077a0162b6d793d2b1dde49effe90933730426e0b`
- lock status immediately before evaluation: `sealed` and verified
- evaluator status: `COMPLETE`

Raw TCGA inputs, cached matrices, participant-level predictions, the fitted
pipeline, and the one-time evaluation receipt are intentionally excluded from
Git. Public documentation reports aggregate results only.

## Interpretation limits

High performance is expected to reflect strong tissue- and cancer-associated
expression differences in a curated retrospective TCGA cohort. It does not
establish prospective clinical utility, robustness to other laboratories or
platforms, independence from all technical or biological confounders, or causal
biomarker status for selected genes. External validation would be required
before making any generalization beyond this study.
