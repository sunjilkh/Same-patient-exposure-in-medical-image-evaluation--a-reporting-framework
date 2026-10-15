NIH FINAL PATCH — PHENOTYPE/VIEW MATCHING + DETERMINISM + VALIDATION-LOCKED THRESHOLDS + CORRECTED XAI

Inputs
- prior executed main-run output: /kaggle/working/_prior_nih_results/nih_corrected_leakage_audit
- raw NIH image root: /kaggle/input

Final primary design
- frozen 6,000-image cohort and fixed rigorous/validation/test manifests reused;
- old predictions/checkpoints/XAI not reused;
- controlled-leak replacements matched exactly on the entire normalized Finding Labels profile + View Position;
- candidate rows without an exact safe profile/view match are excluded rather than creating a composition confound;
- demographics are used as a secondary greedy matching preference and audited;
- strict deterministic paired rerun of ResNet-50 and ConvNeXt-Tiny;
- operating thresholds selected on validation only via Youden J, then locked before test evaluation;
- fixed 0.5 metrics exported separately as supplementary results;
- patient-cluster bootstrap CIs and paired randomization tests;
- corrected ResNet-50 Grad-CAM on a preselected OOF subset;
- XAI quantifies map similarity/centroid shift, not 'border artifact' pathology claims;
- native-prevalence-matched sensitivity pair patched with the same final control design.

Key files
- phenotype_view_matched_leakage_audit.csv
- primary_oof_paired_predictions.csv
- primary_cluster_aware_metrics.csv
- supplementary_fixed_0p5_threshold_metrics.csv
- manuscript_primary_result_table.csv
- native_phenotype_view_matched_audit.csv
- native_paired_predictions.csv
- native_cluster_aware_metrics.csv
- training_summary.csv
- final_integrity_summary.json
- xai/preselected_quantitative_xai_cases.csv
- xai/preselected_visual_cases.csv
- xai/gradcam_paired_map_metrics.csv
- xai/gradcam_cluster_summary.csv
- xai/*overlay PNGs
