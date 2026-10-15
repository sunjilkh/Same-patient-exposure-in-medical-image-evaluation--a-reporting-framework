# Data Availability & Dataset Acquisition Guide

This repository analyzes three benchmark clinical image datasets across three imaging modalities. In accordance with dataset provider governance, data-use agreements, and journal data-availability guidelines, **raw image files and model checkpoints are not redistributed directly within this repository**.

Instead, complete patient-level split manifests, image hashes, metadata identifiers, and out-of-fold prediction matrices are provided in `results/` to enable full verification without requiring multi-gigabyte downloads.

---

## 1. Dataset Acquisition Sources

| Modality / Dataset | Primary Source | Access DOI / URL | Provider License |
| :--- | :--- | :--- | :--- |
| **Chest Radiography**<br>NIH ChestX-ray14 | NIH Clinical Center | [NIH Box Repository](https://nihcc.app.box.com/v/ChestXray-NIHCC) | Public Domain / NIH Terms |
| **Dermoscopy**<br>SIIM-ISIC 2020 | ISIC Archive / Kaggle | [ISIC Challenge 2020](https://challenge2020.isic-archive.com/) | CC BY-NC 4.0 (Non-Commercial) |
| **Retinal OCT**<br>Kermany OCT2017 | Mendeley Data | [doi:10.17632/rscbjbr9sj.2](https://doi.org/10.17632/rscbjbr9sj.2) | CC BY 4.0 |

> [!IMPORTANT]
> **Licensing Compliance:** The SIIM-ISIC 2020 dataset is governed by the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0). Raw image redistribution via public code repositories is prohibited. Researchers must download the images directly from the official challenge portal.

---

## 2. Experimental Cohort Specifications

All cohorts were sampled under strict laboratory constraints to isolate patient-identity leakage:

### A. NIH ChestX-ray14
- **Clinical Task:** Binary detection of Pleural Effusion vs. Effusion-absent.
- **Source Population:** 112,120 frontal chest radiographs from 30,805 unique patients.
- **Sampled Primary Cohort:** 6,000 images (3,000 effusion, 3,000 effusion-absent) across 3,919 patients.
- **Sibling Structure:** Median 2 images/patient (range 1–17). 53% of patients contribute multiple images.
- **Primary Manifest:** `results/nih/cohort_manifest.csv`

### B. SIIM-ISIC 2020 Dermoscopy
- **Clinical Task:** Binary classification of Malignant Melanoma vs. Benign lesions.
- **Source Population:** 33,126 dermoscopic images from 2,056 unique patients.
- **Sampled Primary Cohort:** 1,162 images (581 malignant, 581 benign) across 796 patients.
  - *Note on Deduplication:* All 584 available malignant cases were initially selected alongside 584 benign controls (1,168 total). Exact decoded-pixel fingerprinting (`decoded_fp`) identified and removed 6 exact duplicated images, yielding the final clean 1,162-image cohort.
- **Sibling Structure:** Median 1 image/patient (range 1–6). ~46% of patients contribute multiple images.
- **Primary Manifest:** `results/isic/primary_cohort_manifest.csv`

### C. Kermany OCT2017
- **Clinical Task:** Binary retinal pathology (CNV, DME, DRUSEN) vs. Normal retina.
- **Source Population:** 84,484 spectral-domain optical coherence tomography B-scans.
- **Sampled Primary Cohort:** 1,200 images across 850 patients:
  - 600 Disease cases (stratified across 400 CNV, 100 DME, 100 DRUSEN).
  - 600 Normal cases.
- **Sibling Structure:** Median 1 image/patient (range 1–13). ~41% of patients contribute multiple images.
- **Primary Manifest:** `results/oct/cohort_manifest.csv`

---

## 3. Directory Layout for Local Reproduction from Raw Images

To re-run the notebooks (`notebooks/final/*.ipynb`) from raw source images, organize the downloaded data directories as follows:

```
data/
├── isic2020/
│   ├── train.csv
│   └── train/                        # Directory containing *.jpg images
├── nih_chestxray14/
│   ├── Data_Entry_2017.csv
│   └── images/                       # Directory containing *.png radiographs (001 to 012)
└── kermany_oct2017/
    └── OCT2017/                      # Directory containing train/, test/, val/ folders
```

In the notebooks, set the corresponding root path:
- **ISIC:** Set `IMAGE_ROOT_OVERRIDE = "/path/to/data/isic2020/train"`
- **NIH:** Set `IMAGE_DIR = "/path/to/data/nih_chestxray14/images"`
- **OCT:** Set `OCT_ROOT = "/path/to/data/kermany_oct2017/OCT2017"`

---

## 4. Manifest Integrity and Split Verification

The repository includes pre-partitioned 5-fold cross-validation manifests in `results/{isic,nih,oct}/splits/`:
- `*leak_train.csv`: Training set containing controlled same-patient sibling replacements.
- `*rig_train.csv`: Rigorous training set with strict zero test-patient overlap.
- `*val.csv`: Independent patient-disjoint validation split used for early stopping and threshold selection.
- `*test.csv`: Common test split identical for both arms.

Integrity hashes and patient allocation consistency can be programmatically verified at any time using:
```bash
python scripts/check_635_integrity.py
```
