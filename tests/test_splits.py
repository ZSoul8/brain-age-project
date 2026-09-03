import importlib.util
from pathlib import Path

import numpy as np
import pytest

#verifica che ogni sito appartenga a un solo fold, tutti i partecipanti e fold siano presenti, venga richiesta una configurazione con meno siti che fold e venfa rilevato un sito diviso artificialmente tra due fold

MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "03_create_splits.py"
SPEC = importlib.util.spec_from_file_location("create_splits", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

create_grouped_folds = MODULE.create_grouped_folds
validate_fold_assignment = MODULE.validate_fold_assignment


def test_each_site_is_assigned_to_one_fold_only():
    groups = np.repeat(["site_a", "site_b", "site_c", "site_d"], [4, 3, 5, 2])
    y = np.linspace(8, 30, num=len(groups))

    assignment = create_grouped_folds(y, groups, n_splits=2)

    assert len(assignment) == len(groups)
    for site in np.unique(groups):
        assert len(np.unique(assignment[groups == site])) == 1


def test_every_fold_and_participant_is_present():
    groups = np.repeat(["a", "b", "c", "d", "e"], 3)
    y = np.arange(len(groups), dtype=float)

    assignment = create_grouped_folds(y, groups, n_splits=5)

    assert set(assignment.tolist()) == {0, 1, 2, 3, 4}
    assert np.all(assignment >= 0)


def test_not_enough_sites_is_rejected():
    groups = np.array(["site_a", "site_a", "site_b", "site_b"])
    y = np.arange(4, dtype=float)

    with pytest.raises(ValueError, match="Servono almeno 3 siti"):
        create_grouped_folds(y, groups, n_splits=3)


def test_validation_detects_a_site_split_between_folds():
    groups = np.array(["site_a", "site_a", "site_b", "site_b"])
    invalid_assignment = np.array([0, 1, 0, 1])

    with pytest.raises(ValueError, match="Il sito site_a è stato diviso"):
        validate_fold_assignment(groups, invalid_assignment, n_splits=2)
