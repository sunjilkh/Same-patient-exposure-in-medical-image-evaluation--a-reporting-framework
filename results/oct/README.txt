OCT FINAL PATCH — SUBTYPE MATCHING + DETERMINISTIC RERUN + CORRECTED XAI

Inputs:
- Prior corrected OCT notebook output: /kaggle/working/_prior_oct_results/oct_corrected_leakage_audit
- Raw OCT dataset root: /kaggle/input/datasets/paultimothymooney/kermany2018/OCT2017 

Final design:
- Frozen 1,200-image cohort and fixed test/validation assignments reused from prior corrected run.
- NO prior model predictions/checkpoints/XAI reused.
- Rigorous and controlled-leak arms rerun under deterministic settings.
- Controlled-leak train uses exact Disease-matched swaps:
  NORMAL<->NORMAL, CNV<->CNV, DME<->DME, DRUSEN<->DRUSEN.
- Exact/decoded-content test images forbidden from both arms.
- Rigorous arm has zero train/test patient overlap.
- ResNet-50 and ConvNeXt-Tiny rerun.
- Grad-CAM quantitative BR analysis on all ResNet-50 OOF test images.
- Visual XAI cases selected before predictions with seeded disease-stratified sampling.
- Patient-cluster bootstrap CIs used for performance and XAI effect estimates.

Key outputs:
- subtype_matched_leakage_audit_by_fold.csv
- oof_paired_predictions.csv
- cluster_aware_metrics.csv
- training_summary.csv
- manuscript_primary_result_table.csv
- final_integrity_summary.json
- splits/
- predictions/
- xai/gradcam_border_ratio_long.csv
- xai/gradcam_border_ratio_paired.csv
- xai/gradcam_border_ratio_cluster_summary.csv
- xai/preselected_visual_cases.csv
- xai/*overlay PNGs
