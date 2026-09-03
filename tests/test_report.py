import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "05_make_report.py"
SPEC = importlib.util.spec_from_file_location("make_report", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

compute_cohort_description = MODULE.compute_cohort_description
compute_sex_metrics = MODULE.compute_sex_metrics


def test_cohort_description_counts_participants_and_sex_codes():
    cohort = pd.DataFrame(
        {
            "AGE_AT_SCAN": [10.0, 12.0, 14.0, 16.0],
            "SEX": [1, 1, 2, 2],
            "SITE_ID": ["a", "a", "b", "b"],
            "n_timepoints": [100, 110, 120, 130],
            "func_mean_fd": [0.1, 0.2, 0.3, 0.4],
        }
    )

    description = compute_cohort_description(cohort).iloc[0]

    assert description["n_participants"] == 4
    assert description["n_sites"] == 2
    assert description["sex_1_n"] == 2
    assert description["sex_2_n"] == 2
    assert description["sex_missing_n"] == 0
    assert description["age_mean"] == 13.0


def test_sex_metrics_cover_every_participant_for_each_model():
    rows = []
    age_true = np.array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
    sex = np.array([1, 1, 1, 2, 2, 2])

    for model_index, model_name in enumerate(MODULE.MODEL_ORDER):
        age_pred = age_true + model_index + np.array([0.5, -0.5, 0.2, -0.2, 0.4, -0.4])
        for subject_index in range(len(age_true)):
            rows.append(
                {
                    "subject_id": f"sub-{subject_index}",
                    "model": model_name,
                    "sex": sex[subject_index],
                    "age_true": age_true[subject_index],
                    "age_pred": age_pred[subject_index],
                    "bag": age_pred[subject_index] - age_true[subject_index],
                }
            )

    result = compute_sex_metrics(pd.DataFrame(rows))

    assert len(result) == 6
    assert set(result["sex_code"]) == {"1", "2"}
    assert result.groupby("model")["n"].sum().eq(6).all()
    assert np.isfinite(result[["mae", "rmse", "r2", "mean_bag"]]).all().all()
