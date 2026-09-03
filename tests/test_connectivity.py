import importlib.util
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "src" / "02_build_features.py"
)
SPEC = importlib.util.spec_from_file_location("build_features", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

timeseries_to_connectome_vector = MODULE.timeseries_to_connectome_vector


def test_connectivity_vector_has_expected_shape_and_finite_values():
    rng = np.random.default_rng(42)
    timeseries = rng.normal(size=(30, 4))

    vector = timeseries_to_connectome_vector(timeseries, expected_rois=4)

    assert vector.shape == (6,)
    assert vector.dtype == np.float32
    assert np.isfinite(vector).all()


def test_connectivity_rejects_constant_roi():
    timeseries = np.column_stack(
        [
            np.arange(10, dtype=float),
            np.ones(10, dtype=float),
        ]
    )

    with pytest.raises(ValueError, match="ROI costante"):
        timeseries_to_connectome_vector(timeseries, expected_rois=2)


def test_connectivity_rejects_non_finite_values():
    timeseries = np.arange(20, dtype=float).reshape(10, 2)
    timeseries[0, 0] = np.nan

    with pytest.raises(ValueError, match="NaN o valori infiniti"):
        timeseries_to_connectome_vector(timeseries, expected_rois=2)


def test_connectivity_rejects_wrong_roi_count():
    timeseries = np.arange(30, dtype=float).reshape(10, 3)

    with pytest.raises(ValueError, match="Numero ROI inatteso"):
        timeseries_to_connectome_vector(timeseries, expected_rois=4)
