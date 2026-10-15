#!/usr/bin/env python3
"""
exposure_subset_audit.py

Performs exposure subset and sibling-trend auditing on paired out-of-fold (OOF)
predictions and preselected XAI maps from the matched-replacement experiment.

Outputs:
  - primary_exposure_groups.csv (accuracy and confidence across exposure subgroups)
  - primary_exposure_trends.csv (Spearman rho with patient-cluster bootstrap 95% CIs)
  - xai/gradcam_cluster_summary.csv (cluster-bootstrap summaries of Grad-CAM stability)

Usage:
  python scripts/exposure_subset_audit.py [--results_dir results/isic] [--bootstraps 2000] [--seed 42]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def true_class_conf(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    return np.where(y == 1, p, 1 - p)


def cluster_bootstrap_mean(df, value_col, B=2000, seed=42):
    pats = np.array(sorted(df.Patient_ID.astype(str).unique()))
    byp = {p: df[df.Patient_ID.astype(str) == p] for p in pats}
    rng = np.random.default_rng(seed)
    vals = []

    for _ in range(B):
        samp = rng.choice(pats, len(pats), replace=True)
        x = pd.concat([byp[p] for p in samp], ignore_index=True)
        vals.append(float(x[value_col].mean()))

    return (
        float(df[value_col].mean()),
        float(np.percentile(vals, 2.5)),
        float(np.percentile(vals, 97.5)),
    )


def cluster_bootstrap_group_difference(df, value_col, group_col="exposed", B=2000, seed=542):
    pats = np.array(sorted(df.Patient_ID.astype(str).unique()))
    byp = {p: df[df.Patient_ID.astype(str) == p] for p in pats}
    rng = np.random.default_rng(seed)
    vals = []

    obs = (
        df.loc[df[group_col] == 1, value_col].mean()
        - df.loc[df[group_col] == 0, value_col].mean()
    )

    for _ in range(B):
        samp = rng.choice(pats, len(pats), replace=True)
        x = pd.concat([byp[p] for p in samp], ignore_index=True)
        if (x[group_col] == 1).any() and (x[group_col] == 0).any():
            vals.append(
                float(
                    x.loc[x[group_col] == 1, value_col].mean()
                    - x.loc[x[group_col] == 0, value_col].mean()
                )
            )

    return (
        float(obs),
        float(np.percentile(vals, 2.5)),
        float(np.percentile(vals, 97.5)),
    )


def run_exposure_audit(results_dir: Path, output_dir: Path, bootstraps: int = 2000, seed: int = 42):
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    oof_path = results_dir / "primary_oof_paired_predictions.csv"
    if not oof_path.exists():
        print(f"[!] File not found: {oof_path}")
        return

    print(f"[*] Loading out-of-fold paired predictions from: {oof_path}")
    oof = pd.read_csv(oof_path)

    lesion_id_available = "LesionExposureType" in oof.columns and oof["LesionExposureType"].notna().any()

    exposure_rows = []
    trend_rows = []

    for arch, d in oof.groupby("architecture"):
        z = d.copy()
        z["correct_leak"] = (z.pred_locked_leak == z.Label).astype(int)
        z["true_conf_leak"] = true_class_conf(z.Label, z.prob_leak)
        z["exposed_any"] = (z.n_train_siblings > 0).astype(int)

        # 1. Main patient-level exposure groups
        for ex, g in z.groupby("exposed_any"):
            exposure_rows.append({
                "Architecture": arch,
                "ExposureType": "any_same_patient_sibling",
                "ExposureGroup": int(ex),
                "n_images": len(g),
                "n_patients": g.Patient_ID.nunique(),
                "Accuracy": g.correct_leak.mean(),
                "MeanTrueClassConfidence": g.true_conf_leak.mean(),
                "MeanSiblingCount": g.n_train_siblings.mean(),
            })

        # 2. Lesion-level descriptive groups when available
        if lesion_id_available:
            for typ, g in z.groupby("LesionExposureType"):
                exposure_rows.append({
                    "Architecture": arch,
                    "ExposureType": "lesion_exposure_category",
                    "ExposureGroup": str(typ),
                    "n_images": len(g),
                    "n_patients": g.Patient_ID.nunique(),
                    "Accuracy": g.correct_leak.mean(),
                    "MeanTrueClassConfidence": g.true_conf_leak.mean(),
                    "MeanSiblingCount": g.n_train_siblings.mean(),
                })

        patients = np.array(sorted(z.Patient_ID.astype(str).unique()))
        byp = {p: z[z.Patient_ID.astype(str) == p] for p in patients}

        exposure_vars = [
            "n_train_siblings",
            "n_same_label_siblings",
            "n_opposite_label_siblings",
        ]
        if lesion_id_available:
            exposure_vars += [
                "n_same_lesion_siblings",
                "n_different_lesion_siblings",
            ]

        # 3. Spearman trends with cluster bootstrap
        for ev in exposure_vars:
            if ev not in z.columns:
                continue
            vv = pd.to_numeric(z[ev], errors="coerce")
            if vv.notna().sum() == 0 or vv.nunique(dropna=True) <= 1:
                continue

            rng = np.random.default_rng(seed + 77 + sum(map(ord, ev)))
            boot_rho = []

            for _ in range(bootstraps):
                samp = rng.choice(patients, len(patients), replace=True)
                x = pd.concat([byp[p] for p in samp], ignore_index=True)
                xv = pd.to_numeric(x[ev], errors="coerce")
                ok = xv.notna()

                if ok.sum() > 2 and xv[ok].nunique() > 1:
                    r, _ = spearmanr(xv[ok], x.loc[ok, "correct_leak"])
                    if np.isfinite(r):
                        boot_rho.append(r)

            ok = vv.notna()
            obs, _ = spearmanr(vv[ok], z.loc[ok, "correct_leak"])

            trend_rows.append({
                "Architecture": arch,
                "ExposureVariable": ev,
                "Spearman_rho": obs,
                "Spearman_CI_Lo": (np.percentile(boot_rho, 2.5) if boot_rho else np.nan),
                "Spearman_CI_Hi": (np.percentile(boot_rho, 97.5) if boot_rho else np.nan),
                "BootstrapReplicates": len(boot_rho),
            })

    exposure_df = pd.DataFrame(exposure_rows)
    trends_df = pd.DataFrame(trend_rows)

    out_groups = output_dir / "primary_exposure_groups.csv"
    out_trends = output_dir / "primary_exposure_trends.csv"

    exposure_df.to_csv(out_groups, index=False)
    trends_df.to_csv(out_trends, index=False)

    print(f"\n[+] Exposure Groups saved to: {out_groups}")
    print(exposure_df.to_string(index=False))

    print(f"\n[+] Exposure Trends (Spearman rho + 95% CIs) saved to: {out_trends}")
    print(trends_df.to_string(index=False))

    # 4. Corrected XAI cluster summary if quantitative map metrics exist
    xai_pairs_path = results_dir / "xai" / "gradcam_paired_map_metrics.csv"
    if xai_pairs_path.exists():
        print(f"\n[*] Auditing Grad-CAM paired maps from: {xai_pairs_path}")
        xai_pairs = pd.read_csv(xai_pairs_path)
        xai_rows = []

        metrics_to_check = [
            m for m in ["cam_cosine_similarity", "cam_pearson_r", "cam_centroid_shift"]
            if m in xai_pairs.columns
        ]

        for metric in metrics_to_check:
            q = xai_pairs.dropna(subset=[metric])
            mean, lo, hi = cluster_bootstrap_mean(
                q, metric, B=bootstraps, seed=seed + 900 + sum(map(ord, metric))
            )
            diff, dlo, dhi = cluster_bootstrap_group_difference(
                q, metric, group_col="exposed" if "exposed" in q.columns else "n_train_siblings",
                B=bootstraps, seed=seed + 1200 + sum(map(ord, metric))
            )
            xai_rows.append({
                "Metric": metric,
                "n_images": len(q),
                "n_patients": q.Patient_ID.nunique(),
                "OverallMean": mean,
                "Overall_CI_Lo": lo,
                "Overall_CI_Hi": hi,
                "ExposedMinusUnexposed": diff,
                "ExposureDifference_CI_Lo": dlo,
                "ExposureDifference_CI_Hi": dhi,
            })

        if xai_rows:
            xai_stats = pd.DataFrame(xai_rows)
            xai_out = output_dir / "xai"
            xai_out.mkdir(parents=True, exist_ok=True)
            out_xai = xai_out / "gradcam_cluster_summary.csv"
            xai_stats.to_csv(out_xai, index=False)
            print(f"[+] Grad-CAM Cluster Summary saved to: {out_xai}")
            print(xai_stats.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Recompute exposure groups and Spearman trends with cluster bootstrap."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/isic",
        help="Path to modality results directory containing primary_oof_paired_predictions.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (defaults to results_dir)",
    )
    parser.add_argument(
        "--bootstraps",
        type=int,
        default=2000,
        help="Number of patient-cluster bootstrap iterations (default: 2000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic bootstrap sampling (default: 42)",
    )

    args = parser.parse_args()
    out = Path(args.output_dir if args.output_dir else args.results_dir)
    run_exposure_audit(Path(args.results_dir), out, bootstraps=args.bootstraps, seed=args.seed)


if __name__ == "__main__":
    main()
