SIIM-ISIC 2020 FINAL FULL SAME-PATIENT LEAKAGE AUDIT

Standalone:
- previous notebook output used: NO
- old predictions/checkpoints/XAI reused: NO

Primary cohort:
- images: 1162
- patients: 796
- positive prevalence: 0.500000
- lesion_id available: True
- lesion_id coverage: 100.00%
- folds: 5
- architectures: ResNet50, ConvNeXtTiny

Final control:
- fixed identical test images for rigorous and controlled-leak arms;
- exact and decoded-pixel test content forbidden from both arms;
- rigorous training has zero test-patient overlap;
- controlled-leak replacements preserve target + diagnosis + site + sex exactly;
- nearest-age replacement within the exact matching stratum;
- separate patient-disjoint validation;
- validation-locked Youden-J threshold for thresholded primary metrics;
- fixed-0.5 metrics exported separately;
- patient-cluster bootstrap CIs + paired randomization;
- same-lesion / different-lesion exposure analysis when lesion_id is available;
- corrected ResNet-50 Grad-CAM attribution-stability analysis.

Key files:
- primary_cohort_manifest.csv
- primary_final_leakage_audit_by_fold.csv
- primary_oof_paired_predictions.csv
- primary_cluster_aware_metrics.csv
- supplementary_fixed_0p5_threshold_metrics.csv
- primary_exposure_groups.csv
- primary_exposure_trends.csv
- manuscript_primary_result_table.csv
- training_summary.csv
- final_integrity_summary.json
- splits/
- predictions/
- xai/preselected_quantitative_xai_cases.csv
- xai/preselected_visual_cases.csv
- xai/gradcam_paired_map_metrics.csv
- xai/gradcam_cluster_summary.csv

Capped imbalance sensitivity:
- enabled: True
- benign cap: 3000
- explicitly NOT described as native/natural ISIC prevalence.
