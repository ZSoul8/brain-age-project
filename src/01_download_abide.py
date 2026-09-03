from pathlib import Path

import numpy as np
import pandas as pd
from nilearn.datasets import fetch_abide_pcp


# Cartelle del progetto e cache locale dei dati ABIDE.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path.home() / "brain_age_project_data" / "abide_pcp"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_ROIS = 200
MOVEMENT_COLUMNS = ["func_mean_fd", "func_num_fd", "func_perc_fd"]


def validate_timeseries(timeseries):
    """Restituisce dimensioni e motivi di esclusione di una serie CC200."""
    reasons = []
    n_timepoints = np.nan
    n_rois = np.nan
    n_constant_rois = np.nan

    try:
        values = np.asarray(timeseries, dtype=float)
    except (TypeError, ValueError):
        return n_timepoints, n_rois, n_constant_rois, ["unreadable_timeseries"]

    if values.ndim != 2:
        return n_timepoints, n_rois, n_constant_rois, ["invalid_shape"]

    n_timepoints, n_rois = values.shape

    if n_timepoints < 2:
        reasons.append("too_few_timepoints")

    if n_rois != EXPECTED_ROIS:
        reasons.append("unexpected_roi_count")

    if not np.isfinite(values).all():
        reasons.append("non_finite_values")
    else:
        n_constant_rois = int(np.sum(np.std(values, axis=0) == 0))
        if n_constant_rois > 0:
            reasons.append("constant_roi")

    return n_timepoints, n_rois, n_constant_rois, reasons


# Scarica o recupera dalla cache tutti i controlli disponibili.
abide = fetch_abide_pcp(
    data_dir=DATA_DIR,
    n_subjects=None,
    pipeline="cpac",
    derivatives=["rois_cc200"],
    quality_checked=True,
    band_pass_filtering=True,
    global_signal_regression=False,
    DX_GROUP=2,
)

phenotypic = pd.DataFrame(abide.phenotypic).reset_index(drop=True)
series = list(abide.rois_cc200)

assert len(series) == len(phenotypic), (
    "Serie e metadati non sono allineati"
)

# source_index conserva la posizione originale condivisa da metadati e serie.
phenotypic = phenotypic.copy()
phenotypic.insert(0, "source_index", np.arange(len(phenotypic)))
phenotypic.to_csv(INTERIM_DIR / "phenotypic_full.csv", index=False)

cohort_records = []
exclusion_records = []

for source_index, row in phenotypic.iterrows():
    reasons = []

    age = pd.to_numeric(row.get("AGE_AT_SCAN"), errors="coerce")
    site = row.get("SITE_ID")
    diagnosis = pd.to_numeric(row.get("DX_GROUP"), errors="coerce")

    if pd.isna(age) or not np.isfinite(age):
        reasons.append("missing_age")

    if pd.isna(site) or not str(site).strip():
        reasons.append("missing_site")

    if pd.isna(diagnosis) or int(diagnosis) != 2:
        reasons.append("not_control")

    if source_index >= len(series):
        n_timepoints = np.nan
        n_rois = np.nan
        n_constant_rois = np.nan
        reasons.append("missing_timeseries")
    else:
        (
            n_timepoints,
            n_rois,
            n_constant_rois,
            timeseries_reasons,
        ) = validate_timeseries(series[source_index])
        reasons.extend(timeseries_reasons)

    subject_id = str(row.get("FILE_ID", row.get("SUB_ID", source_index)))
    included = len(reasons) == 0

    exclusion_records.append(
        {
            "source_index": source_index,
            "subject_id": subject_id,
            "SUB_ID": row.get("SUB_ID"),
            "FILE_ID": row.get("FILE_ID"),
            "SITE_ID": site,
            "AGE_AT_SCAN": age,
            "included": included,
            "exclusion_reason": ";".join(reasons),
            "n_timepoints": n_timepoints,
            "n_rois": n_rois,
            "n_constant_rois": n_constant_rois,
        }
    )

    if included:
        cohort_record = {
            "source_index": source_index,
            "subject_id": subject_id,
            "SUB_ID": row.get("SUB_ID"),
            "FILE_ID": row.get("FILE_ID"),
            "SITE_ID": site,
            "AGE_AT_SCAN": float(age),
            "SEX": row.get("SEX"),
            "DX_GROUP": int(diagnosis),
            "n_timepoints": int(n_timepoints),
            "n_rois": int(n_rois),
            "n_constant_rois": int(n_constant_rois),
        }
        for column in MOVEMENT_COLUMNS:
            cohort_record[column] = row.get(column, np.nan)
        cohort_records.append(cohort_record)

cohort = pd.DataFrame(cohort_records)
exclusions = pd.DataFrame(exclusion_records)

assert cohort["source_index"].is_unique
assert cohort["subject_id"].is_unique
assert len(cohort) + int((~exclusions["included"]).sum()) == len(phenotypic)

cohort.to_csv(INTERIM_DIR / "cohort.csv", index=False)
exclusions.to_csv(INTERIM_DIR / "exclusions.csv", index=False)

excluded = exclusions.loc[~exclusions["included"]]

print("Candidati iniziali:", len(phenotypic))
print("Partecipanti inclusi:", len(cohort))
print("Partecipanti esclusi:", len(excluded))
print("\nMotivi di esclusione:")
print(excluded["exclusion_reason"].value_counts().to_string())
print("\nEtà della coorte inclusa:")
print(cohort["AGE_AT_SCAN"].describe().to_string())
print("\nPartecipanti inclusi per sito:")
print(cohort["SITE_ID"].value_counts().sort_index().to_string())
print("\nDistribuzione del sesso registrato:")
print(cohort["SEX"].value_counts(dropna=False).sort_index().to_string())
print("\nCampioni temporali per sito:")
print(
    cohort.groupby("SITE_ID")["n_timepoints"]
    .agg(["count", "min", "median", "max"])
    .to_string()
)
print("\nFile salvati:")
print(INTERIM_DIR / "cohort.csv")
print(INTERIM_DIR / "exclusions.csv")
