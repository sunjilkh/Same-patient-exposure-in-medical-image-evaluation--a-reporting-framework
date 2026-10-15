# Same-patient Exposure in Medical Image Evaluation: A Reporting Framework and Matched-Replacement Audit Across Three Modalities

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](environment/requirements.txt)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](environment/versions.json)
[![Integrity Audit](https://img.shields.io/badge/Audit_Checks-635%2F635_Passed-brightgreen.svg)](scripts/check_635_integrity.py)
[![Primary Metric Rows](https://img.shields.io/badge/Primary_Metrics-60%2F60_Verified-brightgreen.svg)](MAPPING.md)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.XXXXXXX-blue.svg)](https://zenodo.org/)

Official code, data split manifests, and reproducibility audit repository for the manuscript:
> **Sunzil Khandaker**, *"Same-patient exposure in medical image evaluation: a reporting framework and matched-replacement audit across three modalities."* Submitted to *Journal of Medical Systems* (Springer).

---

## 1. Overview & Methodological Framework

In clinical machine learning benchmarks, evaluating models on randomly partitioned data frequently introduces **same-patient exposure (data leakage)**, where sibling images or volumes from patients in the training set appear in test folds. Traditional studies comparing naive random splits with patient-disjoint splits introduce massive composition and test-set confounds by evaluating the two arms on different test data.

This study implements a **matched-replacement experimental audit** on **common, identical test folds**:
- **Common Test Fold:** Rigorous and leaky models are scored on identical fixed test images across 5-fold cross-validation.
- **Strict Disjointness in Rigorous Arm:** Zero patient overlap between training, validation, and test partitions ($0.0\%$ overlap audited).
- **Exact Matched Replacement:** Controlled-leak replacements match target, diagnosis, anatomical site, sex, and age (ISIC); normalized 14-disease profile and view position (NIH); and exact 4-pathology subtypes (OCT).
- **Validation-Locked Thresholds:** Operating thresholds are selected solely on disjoint validation sets via Youden's $J$ and locked before test scoring.
- **Multi-Modal Evidence:** Evaluated across three imaging modalities: **NIH ChestX-ray14** (radiography), **SIIM-ISIC 2020** (dermoscopy), and **Kermany OCT2017** (ophthalmic optical coherence tomography).

```
   ┌────────────────────────────────────────────────────────┐
   │                  Common Test Fold                      │
   │           (Identical Test Images for Both Arms)        │
   └───────────────────┬────────────────┬───────────────────┘
                       │                │
            Evaluated on                Evaluated on
                       │                │
   ┌───────────────────▼────────┐  ┌────▼───────────────────┐
   │     Rigorous Arm           │  │   Controlled-Leak Arm   │
   │  • Zero Test Patients      │  │  • Matched Replacements│
   │  • Disjoint Validation     │  │  • Same Target/Profile │
   │  • Pure Generalization     │  │  • Realized Sibling Mix│
   └────────────────────────────┘  └────────────────────────┘
```

---

## 2. Repository Structure

```
repo/
├── README.md                 # Primary documentation, quickstart, and reproduction guide
├── LICENSE                   # Apache 2.0 open-source license
├── CITATION.cff              # Citation metadata for academic referencing
├── DATA.md                   # Dataset acquisition sources, licenses, and directory layout
├── MAPPING.md                # Exhaustive table/figure/section-to-artifact mapping
├── checksums.sha256          # SHA-256 cryptographic hashes of all tracked files
├── environment/
│   ├── versions.json         # Exact deterministic runtime environment versions
│   ├── requirements.txt      # Python dependencies for pip
│   └── environment.yml       # Conda environment definition
├── notebooks/
│   ├── final/                # Final executed notebooks (all cell outputs preserved)
│   │   ├── isic.ipynb        # Standalone full pipeline for SIIM-ISIC 2020
│   │   ├── nih.ipynb         # Final phenotype/view-matched patch for NIH ChestX-ray14
│   │   └── oct.ipynb         # Final subtype-matched patch for Kermany OCT2017
│   └── provenance/           # Earlier provenance runs (ties to Table S13)
│       ├── nih-x-ray.ipynb   # Main corrected NIH run (established 6,000-image cohort)
│       ├── oct-retina-improved-final.ipynb # Main corrected OCT run (1,200-image cohort)
│       └── PROVENANCE.md     # Provenance timeline, rationale, and correction history
├── results/                  # Complete unzipped experimental artifacts (39.4 MB)
│   ├── isic/                 # splits/, predictions/, xai/, metrics, audit CSVs
│   ├── nih/                  # splits/, predictions/, xai/, metrics, audit CSVs
│   └── oct/                  # splits/, predictions/, xai/, metrics, audit CSVs
└── scripts/
    ├── check_635_integrity.py    # Automated 635-check evidence-consistency test suite
    └── exposure_subset_audit.py  # Sibling exposure trends & cluster bootstrap CIs
```

---

## 3. Quickstart: Instant Verification (No Raw Images Needed)

All split manifests, out-of-fold paired predictions, and metric records are bundled in `results/`. You can independently audit the study without downloading 50+ GB of source images.

### Step 1: Clone the Repository & Setup Environment
```bash
git clone https://github.com/sunjilkh/Same-patient-exposure-in-medical-image-evaluation--a-reporting-framework.git
cd Same-patient-exposure-in-medical-image-evaluation--a-reporting-framework

# Option A: using pip
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r environment/requirements.txt

# Option B: using conda
conda env create -f environment/environment.yml
conda activate split_robustness
```

### Step 2: Run the 635-Check Evidence Consistency Suite
Run the automated assertion suite that validates all 15 primary fold pairs, hash deduplication, patient isolation, and recomputes agreement across all 60 primary metric rows reported in Section 3.1:
```bash
python scripts/check_635_integrity.py
```
**Expected Output:**
```
========================================================================
  FINAL AUDIT SUMMARY
========================================================================
  Total Formal Checks:       635/635 (100.0% Passed)
  Failed Checks:             0
  Primary Metric Rows:       60/60 Verified Agreement (100.0%)
  Audit Status:              PASSED (FULL CONSISTENCY)
========================================================================
```

### Step 3: Run the Exposure Subset Audit
Recompute accuracy across exposure subgroups, lesion-level exposure categories, and Spearman rank correlations with patient-cluster bootstrap 95% confidence intervals (Section 3.4 & Table S7):
```bash
python scripts/exposure_subset_audit.py --results_dir results/isic --bootstraps 2000
```

---

## 4. Re-running Full Training From Scratch (With Raw Images)

To retrain models from ImageNet initialization and reproduce raw predictions:

1. **Acquire Source Datasets:** Follow the download links and licensing instructions in [`DATA.md`](DATA.md).
2. **Execute Final Notebooks:**
   - **ISIC 2020 (`notebooks/final/isic.ipynb`):** Self-contained, one-run execution starting from raw metadata. Expected runtime: ~75 minutes on an NVIDIA Tesla T4 GPU.
   - **NIH ChestX-ray14 (`notebooks/final/nih.ipynb`):** Consumes the frozen primary cohort from `results/nih/cohort_manifest.csv` and retrains paired models under phenotype/view matching. Expected runtime: ~90 minutes on an NVIDIA Tesla T4 GPU.
   - **Kermany OCT2017 (`notebooks/final/oct.ipynb`):** Consumes the frozen primary cohort from `results/oct/cohort_manifest.csv` and retrains paired models under exact subtype matching. Expected runtime: ~45 minutes on an NVIDIA Tesla T4 GPU.

> [!NOTE]
> All final notebooks are provided with **pre-executed cell outputs intact**. Reviewers can inspect every terminal output, training loss log, ROC curve, and Grad-CAM figure directly in Jupyter or GitHub without re-running compute.

---

## 5. Artifact & Manuscript Mapping

See [`MAPPING.md`](MAPPING.md) for the complete bidirectional index:
- **Table 3 (Headline Metrics):** `results/{isic,nih,oct}/primary_cluster_aware_metrics.csv`
- **Table 4 (Exhaustive Metric Panel):** `results/{isic,nih,oct}/manuscript_primary_result_table.csv`
- **Table 5 (McNemar Tests):** Computed from `results/{isic,nih,oct}/` out-of-fold predictions
- **Section 3.1 (Integrity & 635 Checks):** `scripts/check_635_integrity.py`
- **Section 3.4 (Exposure Trends):** `results/isic/primary_exposure_trends.csv`
- **Supplementary Table S13 (Protocol Provenance):** [`notebooks/provenance/PROVENANCE.md`](notebooks/provenance/PROVENANCE.md)

---

## 6. Updating the Manuscript Data & Code Availability Statement

Before final acceptance, update the Data and Code Availability statement in the manuscript text:

```latex
\section*{Data availability}
The NIH ChestX-ray14 dataset is available at \url{https://nihcc.app.box.com/v/ChestXray-NIHCC}.
The SIIM-ISIC 2020 dataset is available at \url{https://challenge2020.isic-archive.com/}.
The Kermany OCT2017 dataset is available at \url{https://doi.org/10.17632/rscbjbr9sj.2}.

\section*{Code availability}
All experimental code, split manifests, saved predictions, audit scripts, and reproducibility materials are openly available in a public GitHub repository at \url{https://github.com/sunjilkh/Same-patient-exposure-in-medical-image-evaluation--a-reporting-framework} and archived persistently on Zenodo with DOI: \url{https://doi.org/10.5281/zenodo.XXXXXXX}.
```

---

## 7. Blinded Peer Review Compliance (For Springer JMS)

If *Journal of Medical Systems* enforces double-blind peer review at your stage:
1. Generate an anonymized repository clone via [Anonymous GitHub](https://anonymous.4open.science/) or Zenodo sandbox.
2. In the anonymized copy, replace author names with `[Anonymous Author]` in `CITATION.cff` and git commits.
3. Submit the anonymous URL in the submission portal. The public GitHub / Zenodo release should be unveiled upon manuscript acceptance.

---

## 8. License & Citation

- **Code & Framework:** Licensed under the [Apache License 2.0](LICENSE).
- **Data Manifests & Predictions:** Provided for research reproduction; subject to underlying dataset terms (ISIC CC BY-NC 4.0; Kermany CC BY 4.0).

Please cite this work using the metadata in [`CITATION.cff`](CITATION.cff).
