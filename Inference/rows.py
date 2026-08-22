"""
Inference/rows.py
Created on July 14, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from Inference.report_utils import (
    report_metric, report_metric_statistic, report_paired_diff,
    report_permutation_2, report_permutation_k,
    report_auroc, report_average_precision, report_paired_auroc_diff,
    report_spearman, report_partial_spearman, report_slope,
    report_mixedlm_slope, report_cramers_v, report_kappa,
    report_hypergeom_overlap, report_stratified_overlap, report_raw_stat,
    report_ece, report_risk_coverage_auc,
)
from Inference.stats_utils import N_BOOT, N_PERM


IDENT_COLS: List[str] = [
    "model",
    "cohort",
    "transform",
    "variant",
    "split",
    "budget_k",
    "budget_frac",
    "fpr",
    "at",
    "conditioning",
    "model_a",
    "model_b",
    "covariate",
    "outcome",
    "baseline",
    "statistic",
    "experiment",
    "stage",
    "analysis",
    "arm",
    "run_id",
    "encoder",
    "seed",
    "pool",
    "site",
    "stratum",
    "finding",
    "pooling",
    "resolution",
    "case_set",
    "subgroup",
    "slice_key",
    "comparison",
    "fdr_family",
]

METRIC_COLS: List[str] = [
    "kind", "readout", "metric", "description", "direction", "unit",
    "mean_pct", "std_pct", "ci_low_pct", "ci_high_pct",
    "mean_raw", "std_raw", "ci_low_raw", "ci_high_raw",
    "n", "n_a", "n_b", "p_raw", "p_fdr", "significant_fdr05",
]

STAT_COLS: List[str] = [
    "kind", "measure", "description", "direction", "unit",
    "value", "value_raw", "r_squared", "intercept_raw", "k_groups",
    "n_groups", "fit_method", "extra",
    "n", "p_raw", "p_fdr", "significant_fdr05",
]

CURVE_COLS: List[str] = IDENT_COLS + [
    "curve_type", "series", "x_name", "x_value", "y_name",
    "y_value", "y_ci_low", "y_ci_high", "y_unit", "n",
]

NAME_MAP: Dict[str, Tuple[str, str, str]] = {
    "selective_precision": (
        "Selective linkage precision: among the k query images the adversary is "
        "most confident about, ranked by the top-1 minus top-2 similarity margin, "
        "the fraction whose top-ranked report is the true one. The chance level is "
        "exactly 1/N for a corpus of N distinct studies, by construction and with "
        "no empirical baseline.",
        "lower_better", "percent"),
    "tpr_at_fpr": (
        "True-positive rate of the linkage attack at a fixed false-positive rate, "
        "sweeping a threshold on the confidence margin. This is the form in which "
        "an average near chance and a confidently exposed minority are told apart.",
        "lower_better", "percent"),
    "recall_at_1": (
        "Fraction of query images whose true report is ranked first in the whole "
        "corpus. Reported for comparison with the retrieval literature; it is an "
        "average-case metric and not the risk quantity.",
        "lower_better", "percent"),
    "recall_at_5": ("Fraction of query images whose true report is in the top 5.",
                    "lower_better", "percent"),
    "recall_at_10": ("Fraction of query images whose true report is in the top 10.",
                     "lower_better", "percent"),
    "mrr": ("Mean reciprocal rank of the true report over the whole corpus.",
            "lower_better", "percent"),
    "recovered_fraction": (
        "Fraction of the cohort whose true report is recovered at rank one at "
        "full-corpus scale. The concentration of the risk, not its average.",
        "lower_better", "percent"),
    "content_retained": (
        "Agreement of the 14 findings extracted from a transformed report against "
        "the original labels, per study and then macro-averaged over findings. The "
        "clinical content a report-side transformation keeps.",
        "higher_better", "percent"),
    "content_concept_recall": (
        "Fraction of the findings the ORIGINAL report asserted that the transformed "
        "report still asserts. Agreement alone is dominated by the many findings a "
        "report never mentions, so this reads the same transformation from the side "
        "a clinician cares about: what was lost, not what stayed absent.",
        "higher_better", "percent"),
    "exposure_feature": (
        "The level of one report covariate, measured over the reports the attack "
        "recovered and over the cohort as a whole. What a successful linkage "
        "actually hands the adversary, and whether the recovered set is adversely "
        "selected. Read the `covariate` column for which one, and `unit` for its "
        "scale: a has_* flag is a percent and a count is raw.",
        "none", "percent"),
    "collapse_rate": (
        "Chance-calibrated neighborhood discordance rate: the fraction of "
        "positives whose mean cosine to the k nearest negatives exceeds the "
        "mean cosine to the k nearest positives, inside one acquisition "
        "stratum, with equal-size banks. Chance is exactly 50.0.",
        "target_50", "percent"),
    "discordant_margin_mean": (
        "Mean continuous discordant margin DM(x) = c_neg(x) - c_pos(x), scaled "
        "by 100. Strictly more informative than its sign, and free.",
        "lower_better", "percent"),
    "discordant_margin_median": (
        "Median continuous discordant margin DM(x), scaled by 100.",
        "lower_better", "percent"),
    "knn_purity": (
        "Fraction of a positive's k nearest neighbors in the combined "
        "equal-size banks that carry its own label.",
        "higher_better", "percent"),
    "silhouette": (
        "Within-stratum silhouette coefficient on the finding label, scaled by "
        "100. Its natural range is [-1, 1], so percent here means the "
        "coefficient times 100, not a proportion.",
        "higher_better", "percent"),
    "low_budget_auroc": (
        "Linear-classifier AUROC fit on a small fraction of the labels.",
        "higher_better", "percent"),

    "task_auroc": ("Linear-classifier AUROC on the finding, the incumbent "
                   "selection metric.", "higher_better", "percent"),
    "task_ap": ("Linear-classifier average precision on the finding.",
                "higher_better", "percent"),
    "retention": (
        "Erasure retention rho = <g(x), g(x with the evidence region "
        "mean-filled)>. Near one means the representation barely moves when the "
        "evidence leaves the image. Uses no label and no neighbors.",
        "lower_better", "percent"),
    "retention_patch": (
        "Erasure retention under a single fixed-size evidence PATCH, the "
        "erasure mode the CEC loss trains on.", "lower_better", "percent"),
    "impurity": (
        "Neighborhood label impurity r(x) = 1 - |2*p_hat_k - 1|, a label-free "
        "failure flag.", "lower_better", "percent"),
    "impurity_auroc": (
        "AUROC of the label-free impurity score against the trained head's own "
        "errors.", "higher_better", "percent"),
    "confidence_auroc": (
        "AUROC of the head's own confidence against its errors, the single "
        "detector the published literature collapses MSP, max-logit and energy "
        "into.", "higher_better", "percent"),

    "noise_floor_seed": (
        "Seed-to-seed standard deviation of the collapse rate on the "
        "matched-control arm. OUR OWN noise floor, which the effect target is "
        "set against, not a literature value.", "lower_better", "percent"),
    "noise_floor_encoder": (
        "Encoder-to-encoder collapse difference between dinov3_b and dinov3_l, "
        "one clean scale step. The second half of our own noise floor.",
        "lower_better", "percentage_points"),

    "effective_rank": (
        "Normalized effective rank of the encoder's embedding matrix, the "
        "entropy of its normalized singular-value spectrum divided by log(d). "
        "Measured before and after adaptation (O1).", "higher_better", "raw"),
    "device_occlusion_drop": (
        "Change in the finding score when the support-device region is "
        "occluded. A model solving pneumothorax by looking at the chest tube "
        "drops a lot.", "lower_better", "percentage_points"),
    "label_noise_rate_reader": (
        "Radiologist-adjudicated rate at which an image's automatic label is "
        "wrong.", "lower_better", "percent"),
    "uncertainty_code_rate": (
        "Rate of the CheXpert labeler's uncertainty code (2) among a set of "
        "negatives. A high rate means the set is enriched for genuine clinical "
        "ambiguity, which is NOT the same as being mislabeled.",
        "none", "percent"),
    "consensus_clean_collapse": (
        "Collapse rate restricted to the subset where all three VinDr-CXR "
        "radiologists agree. Directly annotated, no NLP labeler anywhere in the "
        "chain.", "target_50", "percent"),

    "abc_bound_holds": (
        "Fraction of failed cases where the triangle inequality A >= B - C is "
        "satisfied. It holds by construction, so anything below 100 is a bug.",
        "higher_better", "percent"),
    "abc_a": ("Distance from the positive to its nearest opposite-label "
              "neighbor.", "higher_better", "raw"),
    "abc_b": ("Erasure movement, sqrt(2 - 2*rho).", "higher_better", "raw"),
    "abc_c": ("Distance from the ERASED positive to the negative. Small C means "
              "the negative IS the erased positive geometrically, and the bound "
              "binds tightly; CEC is predicted to work there and NOT to work at "
              "high C.", "none", "raw"),

    "pointing_game": (
        "Does the argmax of the saliency map fall inside the gold mask. The most "
        "robust and least threshold-dependent localization metric, and the "
        "primary one.", "higher_better", "percent"),
    "deletion_auc": (
        "Area under the curve of the score as the highest-saliency patches are "
        "progressively deleted. Threshold-free, and it measures whether the "
        "saliency is causally load-bearing.", "lower_better", "percent"),
    "insertion_auc": (
        "Area under the curve of the score as the highest-saliency patches are "
        "progressively inserted into a blank image.", "higher_better", "percent"),
    "mask_energy": ("Fraction of total saliency mass falling inside the gold "
                    "region.", "higher_better", "percent"),
    "pixel_iou": (
        "Pixel IoU of the thresholded saliency map against the gold "
        "segmentation. A WEAK endpoint: saliency is known to be unreliable, and "
        "the pointing game and the deletion curve are the ones the paper leans "
        "on.", "higher_better", "percent"),

    "failure_set_sensitivity": (
        "Sensitivity on the consensus failure set at the fixed operating point "
        "(sensitivity 80 on validation). THE PAPER'S CLINICAL NUMBER.",
        "higher_better", "percent"),
    "failure_set_miss_rate": (
        "Miss rate on the consensus failure set at the fixed operating point.",
        "lower_better", "percent"),
    "ece": ("Expected calibration error. Items with no confidence score must "
            "NOT sit in the denominator.", "lower_better", "percent"),
    "risk_coverage_auc": ("Area under the risk-coverage curve.",
                          "lower_better", "percent"),
    "auroc_retention_external": (
        "External-site AUROC as a fraction of the within-site oracle head's "
        "AUROC.", "higher_better", "percent"),
    "underdiagnosis_rate": (
        "Rate at which a truly positive case is scored negative at the fixed "
        "operating point, within one subgroup.", "lower_better", "percent"),
    "subgroup_gap": (
        "Largest between-subgroup difference in the metric, in percentage "
        "points.", "lower_better", "percentage_points"),

    "reader_agreement": ("Raw percent agreement between the two readers, "
                         "reported alongside kappa because kappa alone reads "
                         "poorly.", "higher_better", "percent"),
    "forced_choice_rate": (
        "Rate at which the radiologist picks a given arm's overlay as the one "
        "looking at the finding. Chance with three arms is 33.3.",
        "higher_better", "percent"),
    "pair_difficulty": ("Mean 1-to-5 rating of how genuinely hard a failed pair "
                        "is to tell apart.", "higher_better", "percent"),

    "gate_cheap_fix_repairs": (
        "E1 GATE. Does a cheap fix (patch_mean at 224, patch_local, or 448) already "
        "land the panel-median collapse rate within the repair band of chance? 1 "
        "means the paper pivots per the pre-registered rule.", "none", "raw"),
    "noise_floor_pp": (
        "OUR OWN measured noise floor: the encoder-to-encoder collapse difference "
        "between dinov3_b and dinov3_l at cls/224, in percentage points. The effect "
        "target is set against this, not a literature value.",
        "lower_better", "percentage_points"),
    "operator_shuffle_chance": (
        "Collapse rate with the bank LABELS SHUFFLED, in percent. Must land at 50.0 "
        "within tolerance; anything else means the equalization or the "
        "self-exclusion is broken and every collapse number is suspect.",
        "target_50", "percent"),
    "operator_planted_check": (
        "The planted-encoder operator check: 1 when the BLIND encoder lands at "
        "chance AND the SEEING encoder lands at zero. If it fails, the operator "
        "implementation is wrong and nothing else can be trusted.", "none", "raw"),
    "o2_shortcut_prediction_holds": (
        "O2. Do geometrically failed pneumothorax positives lose MORE finding score "
        "under device occlusion than non-failed ones? 1 means shortcut reliance and "
        "geometric collapse are one mechanism.", "none", "raw"),
    "o4a_not_label_artifact": (
        "O4a, THE SECOND GATE. Does the collapse rate on directly-annotated "
        "(non-NLP) labels sit within the band of the comparison rate? 1 means the "
        "premise survives the label-artifact objection.", "none", "raw"),
    "o4c_uncertainty_enrichment": (
        "O4c. The CheXpert uncertainty-code rate among failed-pair negatives, "
        "against the matched control set. Enrichment means the operator selects for "
        "genuine clinical ambiguity, which is NOT the same as mislabeling.",
        "none", "raw"),
    "o4c_matched_groups_built": (
        "O4c wrote the matched failed / control negative groups the E9 reader study "
        "reuses. 1 means the file exists and E9 must not resample its own controls.",
        "none", "raw"),
    "primary_endpoint_supported": (
        "E3 PRIMARY ENDPOINT. Does C2 beat C1 on R1 AND at least three of R2-R5, all "
        "FDR-significant and above the measured noise floor, on a majority of "
        "seeds?", "none", "raw"),
    "secondary_pair_supported": (
        "A SECONDARY robustness pair (MAE, image-text, rad_dino init, DenseNet init, "
        "dinov3-L scale) under the same rule. Reported, never pre-registered as "
        "primary.", "none", "raw"),
    "primary_effect_vs_noise_floor": (
        "The primary R1 effect in percentage points, signed, beside the noise floor "
        "it is judged against. Negative clears the floor, because collapse is "
        "lower-better.", "lower_better", "percentage_points"),
    "e4_falsification_passed": (
        "E4 FALSIFICATION. The P1 placebo must NOT match C2. 1 means the method is "
        "not a fill detector and the paper survives its own strongest attack.",
        "none", "raw"),
    "e4_theory_corroborated": (
        "E4. Does the naive hinge (P4) underperform CEC, as the theory predicts? 1 "
        "means the matched control is doing real work.", "none", "raw"),
    "e4_specificity_held": (
        "E4 SPECIFICITY (P9). CEC must NOT improve the nuisance targets. 1 means the "
        "method is an evidence mechanism rather than a generic regularizer.",
        "none", "raw"),
    "e5_ladder_lowest_rung": (
        "E5. The lowest annotation rung that still works, which IS the deployment "
        "claim. 1 means a rung below full gold boxes carries the effect.",
        "none", "raw"),
    "e5_generalization_holds": (
        "E5. Does the effect survive this generalization stress (box-site holdout, "
        "leave-one-finding-out, budget, silver-vs-gold, box-free, patch size)?",
        "none", "raw"),
    "e6_moves_off_chance": (
        "E6 HEADLINE. Does the CEC arm move this modality off chance, where every "
        "frozen reference encoder sits at 50.0?", "none", "raw"),
    "e6_no_regression_held": (
        "E6. Do the CEC-trained encoders avoid getting worse on the fine-grained "
        "tasks the frozen regime already solves?", "none", "raw"),
    "e7_no_harm_held": (
        "E7. Task AUROC must not degrade under CEC. 1 means the geometric fix costs "
        "no discriminative performance.", "none", "raw"),
    "e7_subgroup_gap_held": (
        "E7. The between-subgroup performance gap did not widen under CEC.",
        "none", "raw"),
    "e7_clinical_miss_rate_fell": (
        "E7 ENDPOINT 8, THE PAPER'S CLINICAL NUMBER. Did the miss rate on the "
        "consensus failure set, at the fixed sensitivity-80 operating point, fall "
        "under CEC relative to the matched baseline? Read it beside the paired "
        "failure_set_miss_rate difference, which carries the CI and the p-value.",
        "none", "raw"),
    "e8_gain_concentrates_low_c": (
        "E8 MECHANISM. Does the CEC gain concentrate in the LOW-C quartiles, where "
        "the erased positive lands essentially on the negative and the bound binds "
        "tightly? This is the theory's central per-case prediction.", "none", "raw"),
    "theory_bound_holds": (
        "E8.5. A synthetic proposition check: 1 when the bound held on every draw of "
        "that seed. A violation means the paper's algebra does not hold on "
        "constructed data.", "none", "raw"),
    "supp_panel_vs_single_gap": (
        "Supplementary. The gap between the consensus panel and a single encoder, in "
        "percentage points.", "none", "percentage_points"),
    "supp_impurity_detector_better": (
        "Supplementary. Does the label-free neighborhood impurity score detect the "
        "trained head's own errors better under CEC?", "none", "raw"),
    "reader_pair_difficulty_mean": (
        "E9. Mean 1-to-5 radiologist rating of how genuinely hard a pair is to tell "
        "apart, on its natural ordinal scale.", "higher_better", "raw"),
    "reader_audit_rate": (
        "E9. Fraction of the reader packet actually adjudicated, so an incomplete "
        "return is never read as a result.", "higher_better", "raw"),
    "stage_manifest": (
        "A recorded stage outcome that produced no computed number: what ran, what "
        "it wrote, or what it deliberately did not compute. Carried so a stage is "
        "never silently absent from the final tables.", "none", "raw"),

    "coverage": ("Fraction of eligible cases that produced a usable value in "
                 "this cell. A cell below the coverage threshold FAILS loudly.",
                 "higher_better", "percent"),
    "exclusion_rate": ("Fraction of cases excluded from this metric, reported "
                       "per cell so an exclusion is never silent.",
                       "lower_better", "percent"),

    "n_corpus": (
        "The number of DISTINCT STUDIES in the candidate corpus, N. The whole "
        "instrument is read against a chance precision of exactly 1/N, so N travels "
        "with every result and is never a swept experimental parameter.",
        "none", "raw"),
    "chance_precision": (
        "1/N, the precision of an adversary who picks queries without the model and "
        "guesses uniformly from the corpus. Definitional, model-free and cohort-free, "
        "which is what makes the fold-enrichment denominator exact.",
        "none", "raw"),
    "fold_over_chance": (
        "Selective precision divided by the definitional chance precision 1/N. The "
        "`at` column says which quantity the fold is taken at, selective or "
        "recall_at_1, and `budget_k` says at which budget.", "lower_better", "raw"),
    "selective_hits": (
        "The COUNT of correct linkages among the k queries the adversary was most "
        "confident about. A rate hides how many patients a named budget exposes.",
        "lower_better", "raw"),
    "tied_at_top_count": (
        "Queries whose top score is an exact tie with a study that is not their own, "
        "which happens when two studies carry identical report text. They are counted "
        "as recovered, the conservative direction for a privacy claim, so the count "
        "travels with every result and the tie-excluded rate is emitted beside it "
        "under case_set='untied'. At T7 the text is generated from the 14 labels and "
        "duplicates are common, so this is what separates recovery from degeneracy.",
        "none", "raw"),
    "singleton_pool_queries": (
        "Queries whose conditioned candidate set holds exactly one study, meaning the "
        "attributes the release publishes identify that study uniquely with no model "
        "involved at all. A margin needs two candidates, so these are reported as a "
        "count and left out of the margin-ranked analysis instead of being scored.",
        "lower_better", "raw"),
    "queries_dropped_below_min_pool": (
        "Queries left out of a conditioned result because their candidate set fell "
        "below the minimum pool size. These are the MOST identifiable queries, so the "
        "count is reported beside every conditioned number and the retained set is "
        "never described as the cohort.", "none", "raw"),
    "recovered_jaccard": (
        "Jaccard overlap of the case sets two models recover at rank one, with an "
        "upper-tail p against independence WITHIN bands of candidate-pool size. Read "
        "`recovered_overlap_n_strata` for how many bands the test pooled over.",
        "none", "raw"),
    "recovered_odds_ratio_mh": (
        "Mantel-Haenszel odds ratio of one model recovering a case given that the "
        "other did, pooled over bands of candidate-pool size, with a "
        "Haldane-Anscombe correction so a zero discordant total gives a finite value. "
        "Agreement between models with unrelated pretraining places the recoverable "
        "structure in the report and not in a checkpoint.", "none", "raw"),
    "recovered_both": (
        "Number of cases both models recover at rank one.", "none", "raw"),
    "recovered_both_expected_stratified": (
        "Cases both models would be expected to recover if each recovered "
        "independently at its own rate WITHIN each band of candidate-pool size. An "
        "unconditional expectation credits shared case difficulty to the models: two "
        "checkpoints at chance both recover the query whose published attributes "
        "leave two candidates.", "none", "raw"),
    "recovered_overlap_n_strata": (
        "Number of candidate-pool-size bands the overlap null pooled over. One means "
        "the pool size does not vary, so the stratified null equals the "
        "unconditional one.", "none", "raw"),
    "n_recovered": (
        "Number of query images whose true report the attack recovered at rank one. "
        "Below ten, the exposure comparison is not reported at all, so a table can "
        "never show a content difference estimated on a handful of reports.",
        "lower_better", "raw"),
    "report_covariate_slope": (
        "Standardized slope of a recoverability outcome on one report covariate, "
        "from a mixed model with a PATIENT random intercept. Read `covariate` for "
        "which covariate, `outcome` for whether the outcome is the continuous margin "
        "or the binary rank-one event, and `fit_method` for whether the random effect "
        "actually fitted or the row fell back to OLS.", "none", "raw"),
    "sign_agreement_across_models": (
        "Fraction of the panel whose slope on this covariate carries the same sign. A "
        "covariate that predicts recoverability in one checkpoint is a fact about that "
        "checkpoint; one that predicts it in models with unrelated pretraining is a "
        "fact about the report.", "higher_better", "raw"),
    "kendall_tau_recall_vs_selective": (
        "Kendall's tau between the ranking of releases by mean Recall@1 and by "
        "selective precision at this budget. Below one, the two quantities order "
        "releases differently and a custodian choosing on the old one can pick the "
        "release that exposes more patients.", "none", "raw"),
    "ranking_inversions": (
        "Number of release PAIRS the two quantities order differently, with the pairs "
        "themselves named in `extra`. A count alone would not say which release a "
        "custodian would misjudge.", "none", "raw"),
    "statistic_undefined_fraction": (
        "Fraction of queries for which this decision rule cannot be evaluated at all. "
        "The ratio rule needs a positive second-best score, and these are cosines. A "
        "rule an adversary can apply to only part of the cohort is a different rule "
        "from one they can apply to all of it, so the share is reported beside every "
        "configuration-study row rather than filled in with a stand-in value.",
        "lower_better", "raw"),

    "spearman_retention_collapse": (
        "Spearman rho of erasure retention against the collapse rate across the "
        "reference panel.", "higher_better", "raw"),
    "spearman_retention_auroc": (
        "Spearman rho of erasure retention against task AUROC.",
        "lower_better", "raw"),
    "spearman_collapse_area": (
        "Spearman rho of the collapse rate against the lesion area fraction. "
        "Near zero is expected; a strong relationship is either real and "
        "interesting or a coordinate bug in the box manifest.", "none", "raw"),
    "slope_box_budget": ("OLS slope of the collapse rate on the number of "
                         "training boxes.", "lower_better", "raw"),
    "slope_patch_size": ("OLS slope of the collapse rate on the erasure patch "
                         "size in pixels.", "none", "raw"),
    "mixedlm_slope_params": (
        "MixedLM slope of the collapse rate on encoder parameter count with a "
        "random intercept for encoder. A plain OLS over pooled rows treats "
        "within-encoder rows as independent and overstates significance.",
        "none", "raw"),
    "failure_set_overlap": (
        "Jaccard overlap of two encoders' failure sets, against the exact "
        "hypergeometric null.", "none", "raw"),
    "kappa_binary": ("Cohen's kappa between the two readers on the binary "
                     "judgment. Two raters, so reader is a FIXED effect: not "
                     "ICC, not Fleiss.", "higher_better", "raw"),
    "kappa_ordinal": ("Quadratic-weighted kappa between the two readers on the "
                      "1-to-5 ordinal rating.", "higher_better", "raw"),
    "cramers_v_stratum": ("Cramer's V between the stratum label and the finding "
                          "label. A stratum that encodes the label would "
                          "contaminate the whole operator.", "lower_better",
                          "raw"),
    "permutation_f": ("Omnibus permutation F statistic across k groups.",
                      "none", "raw"),
    "dose_response_slope": (
        "OLS slope of the collapse rate on the erasure/box dose, the dose-response "
        "trend the mechanism predicts.", "none", "raw"),
    "collapse_vs_noise_rate_slope": (
        "OLS slope of the collapse rate on the injected label-noise rate.",
        "none", "raw"),
    "cec_retention_gain": (
        "Reduction in erasure retention under CEC relative to the control arm, "
        "expected to concentrate in the small-C quartiles.", "higher_better",
        "percent"),
}

NAME_ALIASES: Dict[str, str] = {
    "erasure_retention":        "retention",
    "label_noise_rate":         "label_noise_rate_reader",
    "forced_choice_pick_rate":  "forced_choice_rate",
    "percent_agreement":        "reader_agreement",
    "cohen_kappa":              "kappa_binary",
    "quadratic_weighted_kappa": "kappa_ordinal",
    "collapse_vs_box_budget":   "slope_box_budget",
    "collapse_vs_patch_size":   "slope_patch_size",
    "pointing_game_hit_rate":   "pointing_game",
    "saliency_energy_in_mask":  "mask_energy",
    "saliency_pixel_iou":       "pixel_iou",
}

STAT_MEASURES = {
    "fold_over_chance", "chance_precision", "n_corpus", "selective_hits",
    "tied_at_top_count", "queries_dropped_below_min_pool", "singleton_pool_queries",
    "recovered_jaccard", "recovered_odds_ratio_mh", "recovered_both",
    "recovered_both_expected_stratified", "recovered_overlap_n_strata", "n_recovered",
    "report_covariate_slope", "sign_agreement_across_models",
    "kendall_tau_recall_vs_selective", "ranking_inversions",
    "statistic_undefined_fraction",
    "effective_rank",
    "abc_a", "abc_b", "abc_c",
    "spearman_retention_collapse", "spearman_retention_auroc",
    "spearman_collapse_area",
    "slope_box_budget", "slope_patch_size", "collapse_vs_noise_rate_slope",
    "mixedlm_slope_params",
    "failure_set_overlap",
    "kappa_binary", "kappa_ordinal",
    "cramers_v_stratum",
    "permutation_f",
    "cohen_kappa", "quadratic_weighted_kappa",
    "collapse_vs_box_budget", "collapse_vs_patch_size", "dose_response_slope",
}

READOUT_OF: Dict[str, str] = {
    "collapse_rate": "R1",
    "discordant_margin_mean": "R2",
    "discordant_margin_median": "R2",
    "knn_purity": "R3",
    "silhouette": "R4",
    "low_budget_auroc": "R5",
}

FDR_FAMILY_COLS: Dict[str, List[str]] = {
    "arm_comparison": ["experiment", "arm", "readout"],
    "per_finding":    ["experiment", "finding"],
    "default":        ["experiment", "analysis"],
}


def describe(name: str) -> Tuple[str, str, str]:
    if name in NAME_MAP:
        return NAME_MAP[name]
    if name in NAME_ALIASES:
        return NAME_MAP[NAME_ALIASES[name]]
    return (name.replace("_", " ").capitalize(), "not_specified", "percent")


def _ident(**kw) -> Dict:
    out = {c: "" for c in IDENT_COLS}
    for k, v in kw.items():
        if k in out and v is not None:
            out[k] = v
    return out


def _finish_metric(packed: Dict, prefix: str, name: str, unit: str,
                   ident: Dict) -> Dict:
    if name in STAT_MEASURES:
        raise ValueError(
            f"[rows] '{name}' is in STAT_MEASURES and cannot be emitted as a "
            f"performance metric. A statistical measure carries a raw value and "
            f"a p-value, never a percent and never a confidence interval. Use "
            f"the matching stat emitter."
        )
    desc, direction, default_unit = describe(name)
    row = dict(ident)
    row.update({
        "kind": "metric",
        "readout": READOUT_OF.get(name, ""),
        "metric": name,
        "description": desc,
        "direction": direction,
        "unit": unit or default_unit,
        "mean_pct":    packed.get(f"{prefix}_mean", np.nan),
        "std_pct":     packed.get(f"{prefix}_std", np.nan),
        "ci_low_pct":  packed.get(f"{prefix}_ci_low", np.nan),
        "ci_high_pct": packed.get(f"{prefix}_ci_high", np.nan),
        "mean_raw":    packed.get(f"{prefix}_mean_raw", np.nan),
        "std_raw":     packed.get(f"{prefix}_std_raw", np.nan),
        "ci_low_raw":  packed.get(f"{prefix}_ci_low_raw", np.nan),
        "ci_high_raw": packed.get(f"{prefix}_ci_high_raw", np.nan),
        "n":   packed.get("n", 0),
        "n_a": packed.get("n_a", np.nan),
        "n_b": packed.get("n_b", np.nan),
        "p_raw": packed.get("p_raw", np.nan),
        "p_fdr": np.nan,
        "significant_fdr05": False,
    })
    return row


def _finish_stat(packed: Dict, prefix: str, name: str, ident: Dict,
                 extra: Optional[str] = None) -> Dict:
    desc, direction, _ = describe(name)
    row = dict(ident)
    row.update({
        "kind": "statistic",
        "measure": name,
        "description": desc,
        "direction": direction,
        "unit": "raw",
        "value":     packed.get(f"{prefix}_value", np.nan),
        "value_raw": packed.get(f"{prefix}_value_raw", np.nan),
        "r_squared":     packed.get("r_squared", np.nan),
        "intercept_raw": packed.get("intercept_raw", np.nan),
        "k_groups":      packed.get("k_groups", np.nan),
        "n_groups":      packed.get("n_groups", np.nan),
        "fit_method":    packed.get("fit_method", ""),
        "extra":         extra or "",
        "n": packed.get("n", 0),
        "p_raw": packed.get("p_raw", np.nan),
        "p_fdr": np.nan,
        "significant_fdr05": False,
    })
    return row


def effective_p(row: Dict) -> float:
    for key in ("p_fdr", "p_raw"):
        v = row.get(key)
        if v is not None and np.isfinite(v):
            return float(v)
    return 1.0


def metric(values, *, name: str, cluster_ids=None, is_percent: bool = True,
           n_boot: int = N_BOOT, **ident) -> Dict:
    packed = report_metric(values, prefix="v", is_percent=is_percent,
                           cluster_ids=cluster_ids, n_boot=n_boot)
    return _finish_metric(packed, "v", name, "percent" if is_percent else "raw",
                          _ident(**ident))


def metric_median(values, *, name: str, cluster_ids=None, n_boot: int = N_BOOT,
                  **ident) -> Dict:
    packed = report_metric_statistic(values, np.nanmedian, prefix="v",
                                     is_percent=True, cluster_ids=cluster_ids,
                                     n_boot=n_boot)
    return _finish_metric(packed, "v", name, "percent", _ident(**ident))


def metric_diff(a, b, *, name: str, comparison: str, cluster_ids=None,
                n_boot: int = N_BOOT, **ident) -> Dict:
    packed = report_paired_diff(a, b, prefix="v", is_percent=True,
                                cluster_ids=cluster_ids, n_boot=n_boot)
    ident["comparison"] = comparison
    return _finish_metric(packed, "v", name, "percentage_points", _ident(**ident))


def metric_diff_unpaired(a, b, *, name: str, comparison: str, is_percent: bool = True,
                         n_boot: int = N_BOOT, n_perm: int = N_PERM,
                         **ident) -> Dict:
    packed = report_permutation_2(a, b, prefix="v", is_percent=is_percent,
                                  n_boot=n_boot, n_perm=n_perm)
    ident["comparison"] = comparison
    unit = "percentage_points" if is_percent else "raw"
    return _finish_metric(packed, "v", name, unit, _ident(**ident))


def auroc(y_true, y_score, *, name: str = "task_auroc", cluster_ids=None,
          n_boot: int = N_BOOT, **ident) -> Dict:
    packed = report_auroc(y_true, y_score, prefix="v", cluster_ids=cluster_ids,
                          n_boot=n_boot)
    return _finish_metric(packed, "v", name, "percent", _ident(**ident))


def average_precision(y_true, y_score, *, name: str = "task_ap", cluster_ids=None,
                      n_boot: int = N_BOOT, **ident) -> Dict:
    packed = report_average_precision(y_true, y_score, prefix="v",
                                      cluster_ids=cluster_ids, n_boot=n_boot)
    return _finish_metric(packed, "v", name, "percent", _ident(**ident))


def auroc_diff(y_true, score_a, score_b, *, name: str, comparison: str,
               cluster_ids=None, n_boot: int = N_BOOT, **ident) -> Dict:
    packed = report_paired_auroc_diff(y_true, score_a, score_b, prefix="v",
                                      cluster_ids=cluster_ids, n_boot=n_boot)
    ident["comparison"] = comparison
    return _finish_metric(packed, "v", name, "percentage_points", _ident(**ident))


def ece(y_true, prob, *, name: str = "ece", cluster_ids=None,
        n_boot: int = N_BOOT, **ident) -> Dict:
    packed = report_ece(y_true, prob, prefix="v", cluster_ids=cluster_ids,
                        n_boot=n_boot)
    return _finish_metric(packed, "v", name, "percent", _ident(**ident))


def risk_coverage_auc(y_true, prob, *, name: str = "risk_coverage_auc",
                      cluster_ids=None, n_boot: int = N_BOOT, **ident) -> Dict:
    packed = report_risk_coverage_auc(y_true, prob, prefix="v",
                                      cluster_ids=cluster_ids, n_boot=n_boot)
    return _finish_metric(packed, "v", name, "percent", _ident(**ident))


def spearman(x, y, *, name: str, **ident) -> Dict:
    return _finish_stat(report_spearman(x, y, prefix="v"), "v", name, _ident(**ident))


def partial_spearman(x, y, z, *, name: str, control: str = "", **ident) -> Dict:
    return _finish_stat(report_partial_spearman(x, y, z, prefix="v"), "v", name,
                        _ident(**ident), extra=f"controlling={control}")


def slope(x, y, *, name: str, **ident) -> Dict:
    return _finish_stat(report_slope(x, y, prefix="v"), "v", name, _ident(**ident))


def mixedlm_slope(x, y, groups, *, name: str, **ident) -> Dict:
    packed = report_mixedlm_slope(x, y, groups, prefix="v")
    cross = packed.get("v_ols_crosscheck", np.nan)
    return _finish_stat(packed, "v", name, _ident(**ident),
                        extra=f"ols_crosscheck_slope={cross}")


def cramers_v(a, b, *, name: str, **ident) -> Dict:
    return _finish_stat(report_cramers_v(a, b, prefix="v"), "v", name, _ident(**ident))


def kappa(rater_a, rater_b, *, name: str, weights: Optional[str] = None,
          n_perm: int = N_PERM, **ident) -> Dict:
    packed = report_kappa(rater_a, rater_b, prefix="v", weights=weights,
                          n_perm=n_perm)
    return _finish_stat(packed, "v", name, _ident(**ident),
                        extra=f"weights={packed.get('kappa_weights', 'unweighted')}")


def overlap(mask_a, mask_b, *, name: str = "failure_set_overlap",
            comparison: str = "", **ident) -> Dict:
    packed = report_hypergeom_overlap(mask_a, mask_b, prefix="v")
    ident["comparison"] = comparison
    excess = packed.get("v_excess", np.nan)
    null_v = packed.get("v_null_value", np.nan)
    return _finish_stat(packed, "v", name, _ident(**ident),
                        extra=f"null_jaccard={null_v};excess={excess}")


def stratified_overlap(mask_a, mask_b, strata_value, *, name: str = "recovered_jaccard",
                       comparison: str = "", n_strata: int = 10, n_perm: int = N_PERM,
                       **ident) -> Dict:
    packed = report_stratified_overlap(mask_a, mask_b, strata_value, prefix="v",
                                       n_strata=n_strata, n_perm=n_perm)
    ident["comparison"] = comparison
    row = _finish_stat(packed, "v", name, _ident(**ident),
                       extra=f"null_jaccard={packed.get('v_null_value')};"
                             f"excess={packed.get('v_excess')};"
                             f"strata={packed.get('v_n_strata')}")
    row["_parts"] = {"expected": packed.get("v_expected"),
                     "odds_ratio_mh": packed.get("v_odds_ratio_mh"),
                     "n_strata": packed.get("v_n_strata"),
                     "p_raw": packed.get("p_raw")}
    return row


def permutation_k(groups, *, name: str = "permutation_f", n_perm: int = N_PERM,
                  **ident) -> Dict:
    packed = report_permutation_k(groups, prefix="v", n_perm=n_perm)
    return _finish_stat(packed, "v", name, _ident(**ident))


def raw_stat(value, *, name: str, n=None, p_raw=None, extra: str = "", **ident) -> Dict:
    packed = report_raw_stat(value, prefix="v", n=n, p_raw=p_raw)
    return _finish_stat(packed, "v", name, _ident(**ident), extra=extra)


VERDICT_NAMES = {k for k in NAME_MAP
                 if k.startswith(("gate_", "o2_", "o4a_", "o4c_", "e4_", "e5_",
                                  "e6_", "e7_", "e8_", "primary_", "secondary_",
                                  "operator_", "theory_", "supp_", "reader_"))
                 or k in ("noise_floor_pp", "stage_manifest")}


def verdict(value, *, name: str, n=None, extra: str = "", **ident) -> Dict:
    if name not in VERDICT_NAMES:
        raise ValueError(
            f"[rows] '{name}' is not a registered verdict name. Add it to NAME_MAP "
            f"with a description and a direction before emitting it, so the final "
            f"tables stay self-describing. Registered: {sorted(VERDICT_NAMES)}")
    return raw_stat(value, name=name, n=n, extra=extra, **ident)


def curve_point(*, curve_type: str, x_name: str, x_value, y_name: str, y_value,
                series: str = "", y_ci_low=np.nan, y_ci_high=np.nan,
                y_unit: str = "percent", n=0, **ident) -> Dict:
    row = _ident(**ident)
    row.update({
        "curve_type": curve_type, "series": series,
        "x_name": x_name, "x_value": x_value,
        "y_name": y_name, "y_value": y_value,
        "y_ci_low": y_ci_low, "y_ci_high": y_ci_high,
        "y_unit": y_unit, "n": n,
    })
    return row


def _ordered(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    if kind == "curve":
        cols = CURVE_COLS
    else:
        cols = IDENT_COLS + (METRIC_COLS if kind == "metric" else STAT_COLS)
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan if c not in IDENT_COLS else ""
    extra = [c for c in df.columns if c not in cols]
    return df[cols + extra]


def write_rows(rows: List[dict], path: str, kind: str = "auto") -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=IDENT_COLS + METRIC_COLS)
    elif kind == "auto":
        df = _ordered(df, "curve" if "curve_type" in df.columns else "metric")
    else:
        df = _ordered(df, kind)
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)
    return path


def append_status(results_dir: str, stage: str, message: str) -> None:
    import time
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"{stage}_status.txt")
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "a") as f:
        f.write(f"[{stamp}] {message}\n")
        f.flush()
        os.fsync(f.fileno())
