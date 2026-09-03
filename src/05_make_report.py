import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_PATH = (
    PROJECT_ROOT / "results" / "predictions" / "predictions.csv"
)
COHORT_PATH = PROJECT_ROOT / "data" / "interim" / "cohort.csv"
EXCLUSIONS_PATH = PROJECT_ROOT / "data" / "interim" / "exclusions.csv"
FOLDS_PATH = PROJECT_ROOT / "data" / "interim" / "outer_folds.csv"
FOLD_SUMMARY_PATH = (
    PROJECT_ROOT / "data" / "interim" / "outer_fold_summary.csv"
)
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features_cc200.npz"
CONFIG_PATH = PROJECT_ROOT / "run_config.json"
VERSIONS_PATH = PROJECT_ROOT / "versions.txt"
REQUIREMENTS_PATH = PROJECT_ROOT / "requirements.txt"

TABLES_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
LOGS_DIR = PROJECT_ROOT / "results" / "logs"
FINAL_DIR = PROJECT_ROOT / "results" / "final"

MODEL_ORDER = ["dummy_median", "ridge", "mlp"]
MODEL_LABELS = {
    "dummy_median": "Dummy mediano",
    "ridge": "Ridge",
    "mlp": "MLP",
}
MODEL_COLORS = {
    "dummy_median": "#6b7280",
    "ridge": "#2563eb",
    "mlp": "#ea580c",
}
MODEL_MARKERS = {
    "dummy_median": "o",
    "ridge": "s",
    "mlp": "^",
}
MIN_SITE_SIZE = 10
FIGURE_DPI = 300


def save_csv_atomically(dataframe, destination):
    """Salva una tabella completa prima di sostituire il CSV precedente."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(".tmp.csv")
    dataframe.to_csv(temporary_path, index=False)
    temporary_path.replace(destination)


def save_json_atomically(data, destination):
    """Salva un JSON completo prima di sostituire il file precedente."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(".tmp.json")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=2, ensure_ascii=False)
    temporary_path.replace(destination)


def save_figure(fig, filename):
    """Salva una figura ad alta risoluzione e chiude la memoria grafica."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    destination = FIGURES_DIR / filename
    fig.savefig(destination, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return destination


def safe_pearson(x, y):
    """Calcola Pearson quando esistono abbastanza valori non costanti."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    x_is_constant = len(x) > 0 and np.allclose(
        x,
        x[0],
        rtol=0,
        atol=1e-12,
    )
    y_is_constant = len(y) > 0 and np.allclose(
        y,
        y[0],
        rtol=0,
        atol=1e-12,
    )
    if len(x) < 3 or x_is_constant or y_is_constant:
        return np.nan, np.nan, len(x)

    result = pearsonr(x, y)
    return float(result.statistic), float(result.pvalue), len(x)


def regression_metrics(y_true, y_pred):
    """Restituisce le metriche principali di regressione."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true e y_pred hanno forme diverse")

    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("Le età vere o predette contengono valori non finiti")

    correlation, correlation_p, _ = safe_pearson(y_true, y_pred)
    return {
        "n": len(y_true),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "pearson_r": correlation,
        "pearson_p": correlation_p,
    }


def load_and_validate_inputs():
    """Carica gli output precedenti e verifica coerenza e completezza."""
    required_paths = (
        PREDICTIONS_PATH,
        COHORT_PATH,
        EXCLUSIONS_PATH,
        FOLDS_PATH,
        FOLD_SUMMARY_PATH,
        FEATURES_PATH,
        CONFIG_PATH,
    )
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"File necessario non trovato: {path}")

    predictions = pd.read_csv(PREDICTIONS_PATH)
    cohort = pd.read_csv(COHORT_PATH)
    exclusions = pd.read_csv(EXCLUSIONS_PATH)
    folds = pd.read_csv(FOLDS_PATH)

    expected_prediction_columns = {
        "subject_id",
        "site",
        "outer_fold",
        "model",
        "age_true",
        "age_pred",
        "bag",
        "absolute_error",
        "sex",
        "func_mean_fd",
    }
    missing_columns = expected_prediction_columns - set(predictions.columns)
    if missing_columns:
        raise RuntimeError(
            f"predictions.csv non contiene {sorted(missing_columns)}"
        )

    if cohort["subject_id"].duplicated().any():
        raise RuntimeError("cohort.csv contiene ID duplicati")

    if set(predictions["model"].unique()) != set(MODEL_ORDER):
        raise RuntimeError("predictions.csv non contiene i tre modelli attesi")

    numeric_prediction_columns = [
        "age_true",
        "age_pred",
        "bag",
        "absolute_error",
    ]
    if not np.isfinite(
        predictions[numeric_prediction_columns].to_numpy(dtype=float)
    ).all():
        raise RuntimeError("Le predizioni contengono valori non finiti")

    if predictions.duplicated(["model", "subject_id"]).any():
        raise RuntimeError("Esistono predizioni duplicate")

    expected_ids = set(cohort["subject_id"].astype(str))
    reference_folds = folds.set_index("subject_id")["outer_fold"].to_dict()

    for model_name in MODEL_ORDER:
        model_rows = predictions.loc[predictions["model"] == model_name]
        if len(model_rows) != len(cohort):
            raise RuntimeError(f"Numero errato di predizioni per {model_name}")
        if set(model_rows["subject_id"].astype(str)) != expected_ids:
            raise RuntimeError(f"ID incompleti per {model_name}")
        observed_folds = model_rows.set_index("subject_id")["outer_fold"].to_dict()
        if observed_folds != reference_folds:
            raise RuntimeError(f"Fold non allineati per {model_name}")

    with np.load(FEATURES_PATH, allow_pickle=False) as features:
        X = features["X"]
        y = features["y"]
        feature_ids = features["subject_ids"].astype(str)
        feature_groups = features["groups"].astype(str)
        if X.shape != (len(cohort), 19900):
            raise RuntimeError(f"Forma inattesa di X: {X.shape}")
        if len(y) != len(cohort) or not np.isfinite(X).all() or not np.isfinite(y).all():
            raise RuntimeError("Feature o target non validi")
        if not np.array_equal(feature_ids, cohort["subject_id"].astype(str)):
            raise RuntimeError("Gli ID delle feature non coincidono con la coorte")
        if not np.array_equal(feature_groups, cohort["SITE_ID"].astype(str)):
            raise RuntimeError("I siti delle feature non coincidono con la coorte")

    excluded_rows = exclusions.loc[~exclusions["included"]]
    if excluded_rows["exclusion_reason"].fillna("").str.strip().eq("").any():
        raise RuntimeError("Esiste un'esclusione senza motivo")

    site_fold_counts = folds.groupby("site")["outer_fold"].nunique()
    if not site_fold_counts.eq(1).all():
        raise RuntimeError("Almeno un sito è stato diviso tra più fold")

    return predictions, cohort, exclusions, folds


def compute_metric_tables(predictions):
    """Calcola metriche complessive, per fold e per sito."""
    overall_records = []
    fold_records = []
    site_records = []

    for model_name in MODEL_ORDER:
        model_rows = predictions.loc[predictions["model"] == model_name]
        overall_records.append(
            {
                "model": model_name,
                **regression_metrics(model_rows["age_true"], model_rows["age_pred"]),
            }
        )

        for fold, fold_rows in model_rows.groupby("outer_fold", sort=True):
            fold_records.append(
                {
                    "model": model_name,
                    "outer_fold": int(fold),
                    **regression_metrics(
                        fold_rows["age_true"],
                        fold_rows["age_pred"],
                    ),
                }
            )

        for site, site_rows in model_rows.groupby("site", sort=True):
            if len(site_rows) < MIN_SITE_SIZE:
                continue
            site_records.append(
                {
                    "model": model_name,
                    "site": site,
                    **regression_metrics(
                        site_rows["age_true"],
                        site_rows["age_pred"],
                    ),
                }
            )

    overall = pd.DataFrame(overall_records)
    overall["model_order"] = overall["model"].map(
        {name: index for index, name in enumerate(MODEL_ORDER)}
    )
    overall = overall.sort_values("model_order").drop(columns="model_order")
    by_fold = pd.DataFrame(fold_records)
    by_site = pd.DataFrame(site_records)
    return overall, by_fold, by_site


def compute_bag_and_confound_tables(predictions):
    """Calcola age bias, calibrazione e associazione errore-movimento."""
    bag_records = []
    movement_records = []

    for model_name in MODEL_ORDER:
        rows = predictions.loc[predictions["model"] == model_name]
        age = rows["age_true"].to_numpy(dtype=float)
        age_pred = rows["age_pred"].to_numpy(dtype=float)
        bag = rows["bag"].to_numpy(dtype=float)
        bag_r, bag_p, bag_n = safe_pearson(age, bag)
        slope, intercept = np.polyfit(age, age_pred, deg=1)

        bag_records.append(
            {
                "model": model_name,
                "n": len(rows),
                "mean_bag": float(np.mean(bag)),
                "std_bag": float(np.std(bag, ddof=1)),
                "bag_age_r": bag_r,
                "bag_age_p": bag_p,
                "calibration_intercept": float(intercept),
                "calibration_slope": float(slope),
            }
        )

        movement = pd.to_numeric(rows["func_mean_fd"], errors="coerce").to_numpy()
        error = rows["absolute_error"].to_numpy(dtype=float)
        movement_r, movement_p, movement_n = safe_pearson(movement, error)
        movement_records.append(
            {
                "model": model_name,
                "n_valid": movement_n,
                "absolute_error_mean_fd_r": movement_r,
                "absolute_error_mean_fd_p": movement_p,
            }
        )

    return pd.DataFrame(bag_records), pd.DataFrame(movement_records)


def compute_age_band_metrics(predictions):
    """Calcola la MAE in fasce d'età definite prima del confronto."""
    rows = predictions.copy()
    rows["age_band"] = pd.cut(
        rows["age_true"],
        bins=[0, 12, 18, 30, np.inf],
        labels=["<12", "12-17", "18-29", "30+"],
        right=False,
    )
    records = []
    for (model_name, age_band), band_rows in rows.groupby(
        ["model", "age_band"],
        observed=True,
        sort=False,
    ):
        records.append(
            {
                "model": model_name,
                "age_band": str(age_band),
                "n": len(band_rows),
                "mae": float(band_rows["absolute_error"].mean()),
            }
        )
    return pd.DataFrame(records)


def compute_cohort_description(cohort):
    """Riassume numerosità, età, sesso, movimento e durata delle scansioni."""
    required_columns = {
        "AGE_AT_SCAN",
        "SEX",
        "SITE_ID",
        "n_timepoints",
        "func_mean_fd",
    }
    missing_columns = required_columns - set(cohort.columns)
    if missing_columns:
        raise RuntimeError(
            f"cohort.csv non contiene {sorted(missing_columns)}"
        )

    ages = pd.to_numeric(cohort["AGE_AT_SCAN"], errors="coerce")
    timepoints = pd.to_numeric(cohort["n_timepoints"], errors="coerce")
    sex = pd.to_numeric(cohort["SEX"], errors="coerce")
    movement = pd.to_numeric(cohort["func_mean_fd"], errors="coerce")

    if ages.isna().any() or timepoints.isna().any():
        raise RuntimeError("Età o numerosità temporali non valide nella coorte")

    description = pd.DataFrame(
        [
            {
                "n_participants": len(cohort),
                "n_sites": cohort["SITE_ID"].nunique(),
                "age_mean": float(ages.mean()),
                "age_std": float(ages.std(ddof=1)),
                "age_median": float(ages.median()),
                "age_min": float(ages.min()),
                "age_max": float(ages.max()),
                "sex_1_n": int(sex.eq(1).sum()),
                "sex_1_percent": float(100 * sex.eq(1).mean()),
                "sex_2_n": int(sex.eq(2).sum()),
                "sex_2_percent": float(100 * sex.eq(2).mean()),
                "sex_missing_n": int(sex.isna().sum()),
                "timepoints_min": int(timepoints.min()),
                "timepoints_median": float(timepoints.median()),
                "timepoints_max": int(timepoints.max()),
                "movement_mean_fd_available_n": int(movement.notna().sum()),
            }
        ]
    )
    return description


def compute_sex_metrics(predictions):
    """Calcola metriche descrittive per il codice di sesso registrato."""
    records = []

    for model_name in MODEL_ORDER:
        model_rows = predictions.loc[predictions["model"] == model_name]
        for sex_value, rows in model_rows.groupby("sex", dropna=False, sort=True):
            if pd.isna(sex_value):
                sex_code = "missing"
            else:
                numeric_sex = float(sex_value)
                sex_code = (
                    str(int(numeric_sex))
                    if numeric_sex.is_integer()
                    else str(numeric_sex)
                )

            metrics = regression_metrics(rows["age_true"], rows["age_pred"])
            bag = rows["bag"].to_numpy(dtype=float)
            records.append(
                {
                    "model": model_name,
                    "sex_code": sex_code,
                    "n": len(rows),
                    "age_mean": float(rows["age_true"].mean()),
                    "age_std": float(rows["age_true"].std(ddof=1)),
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "r2": metrics["r2"],
                    "pearson_r": metrics["pearson_r"],
                    "pearson_p": metrics["pearson_p"],
                    "mean_bag": float(np.mean(bag)),
                    "std_bag": float(np.std(bag, ddof=1)),
                }
            )

    result = pd.DataFrame(records)
    counts = result.groupby("model")["n"].sum()
    if not counts.reindex(MODEL_ORDER).eq(
        predictions["subject_id"].nunique()
    ).all():
        raise RuntimeError("L'analisi per sesso non copre tutti i partecipanti")
    return result


def make_cohort_flow_figure(exclusions):
    """Disegna il flusso candidati-esclusi-inclusi."""
    n_candidates = len(exclusions)
    n_included = int(exclusions["included"].sum())
    excluded = exclusions.loc[~exclusions["included"]]
    n_excluded = len(excluded)
    reason_counts = excluded["exclusion_reason"].value_counts()

    flow_table = pd.DataFrame(
        [
            {"stage": "Candidati ABIDE", "n": n_candidates},
            {"stage": "Esclusi", "n": n_excluded},
            {"stage": "Inclusi", "n": n_included},
        ]
    )
    save_csv_atomically(flow_table, TABLES_DIR / "cohort_flow.csv")

    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    boxes = [
        (0.4, 1.35, 2.3, 1.25, "Candidati ABIDE", n_candidates, "#dbeafe"),
        (3.85, 2.35, 2.3, 1.1, "Esclusi", n_excluded, "#fee2e2"),
        (7.3, 1.35, 2.3, 1.25, "Inclusi", n_included, "#dcfce7"),
    ]
    for x, y, width, height, label, value, color in boxes:
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.08",
            facecolor=color,
            edgecolor="#374151",
            linewidth=1.2,
        )
        ax.add_patch(patch)
        ax.text(x + width / 2, y + height * 0.62, label, ha="center", va="center", fontsize=12)
        ax.text(x + width / 2, y + height * 0.28, f"n = {value}", ha="center", va="center", fontsize=13, fontweight="bold")

    ax.annotate("", xy=(3.85, 2.75), xytext=(2.7, 2.25), arrowprops={"arrowstyle": "->", "color": "#374151", "lw": 1.5})
    ax.annotate("", xy=(7.3, 1.95), xytext=(2.7, 1.95), arrowprops={"arrowstyle": "->", "color": "#374151", "lw": 1.5})
    reason_text = "\n".join(
        f"{reason.replace('_', ' ')}: {count}"
        for reason, count in reason_counts.items()
    )
    ax.text(
        3.2,
        3.08,
        reason_text,
        ha="center",
        va="bottom",
        fontsize=9,
        color="#4b5563",
    )
    ax.text(
        5.0,
        1.78,
        "Superano i controlli",
        ha="center",
        va="top",
        fontsize=9,
        color="#4b5563",
    )
    ax.set_title("Flusso di selezione della coorte", fontsize=14, pad=8)
    return save_figure(fig, "cohort_flow.png")


def make_age_distribution_figure(cohort):
    """Disegna e salva la distribuzione dell'età della coorte."""
    ages = cohort["AGE_AT_SCAN"].to_numpy(dtype=float)
    lower = 2 * np.floor(ages.min() / 2)
    upper = 2 * np.ceil(ages.max() / 2) + 2
    bins = np.arange(lower, upper + 0.01, 2)
    counts, edges = np.histogram(ages, bins=bins)
    histogram_table = pd.DataFrame(
        {"age_start": edges[:-1], "age_end": edges[1:], "count": counts}
    )
    save_csv_atomically(histogram_table, TABLES_DIR / "age_histogram.csv")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(ages, bins=bins, color="#2563eb", edgecolor="white", linewidth=0.8)
    ax.axvline(np.mean(ages), color="#b91c1c", linestyle="--", linewidth=1.5, label=f"Media = {np.mean(ages):.1f}")
    ax.set_title("Distribuzione dell'età nella coorte inclusa")
    ax.set_xlabel("Età cronologica (anni)")
    ax.set_ylabel("Numero di partecipanti")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    return save_figure(fig, "age_distribution.png")


def make_actual_vs_predicted_figure(predictions, overall):
    """Confronta età reale e predetta nei tre modelli."""
    all_values = np.concatenate(
        [predictions["age_true"].to_numpy(), predictions["age_pred"].to_numpy()]
    )
    lower = float(np.floor(np.min(all_values)) - 1)
    upper = float(np.ceil(np.max(all_values)) + 1)
    metrics_lookup = overall.set_index("model")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True)
    for ax, model_name in zip(axes, MODEL_ORDER):
        rows = predictions.loc[predictions["model"] == model_name]
        color = MODEL_COLORS[model_name]
        ax.scatter(rows["age_true"], rows["age_pred"], s=20, alpha=0.58, color=color, edgecolors="none")
        ax.plot([lower, upper], [lower, upper], color="#111827", linestyle="--", linewidth=1.2, label="Predizione perfetta")
        slope, intercept = np.polyfit(rows["age_true"], rows["age_pred"], 1)
        x_line = np.array([lower, upper])
        ax.plot(x_line, slope * x_line + intercept, color=color, linewidth=2, label="Retta osservata")
        metric = metrics_lookup.loc[model_name]
        ax.text(0.04, 0.96, f"MAE = {metric['mae']:.2f}\nR² = {metric['r2']:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=10, bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
        ax.set_title(MODEL_LABELS[model_name])
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.grid(alpha=0.2)
        ax.set_xlabel("Età reale (anni)")
    axes[0].set_ylabel("Età predetta (anni)")
    axes[-1].legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle("Età reale ed età predetta out-of-fold", fontsize=15)
    fig.tight_layout()
    return save_figure(fig, "actual_vs_predicted.png")


def make_bag_figure(predictions, bag_analysis):
    """Mostra il brain-age gap rispetto all'età reale."""
    bag_lookup = bag_analysis.set_index("model")
    y_limit = float(np.ceil(np.max(np.abs(predictions["bag"])))) + 1

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True)
    for ax, model_name in zip(axes, MODEL_ORDER):
        rows = predictions.loc[predictions["model"] == model_name]
        color = MODEL_COLORS[model_name]
        ax.scatter(rows["age_true"], rows["bag"], s=20, alpha=0.55, color=color, edgecolors="none")
        slope, intercept = np.polyfit(rows["age_true"], rows["bag"], 1)
        x_line = np.array([rows["age_true"].min(), rows["age_true"].max()])
        ax.plot(x_line, slope * x_line + intercept, color=color, linewidth=2)
        ax.axhline(0, color="#111827", linestyle="--", linewidth=1.2)
        bag_row = bag_lookup.loc[model_name]
        ax.text(0.04, 0.96, f"r(BAG, età) = {bag_row['bag_age_r']:.2f}\nBAG medio = {bag_row['mean_bag']:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=10, bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
        ax.set_title(MODEL_LABELS[model_name])
        ax.set_ylim(-y_limit, y_limit)
        ax.set_xlabel("Età reale (anni)")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Brain-age gap (anni)")
    fig.suptitle("Brain-age gap rispetto all'età reale", fontsize=15)
    fig.tight_layout()
    return save_figure(fig, "bag_vs_age.png")


def make_model_mae_figure(overall, by_fold):
    """Confronta la MAE complessiva e la variabilità tra fold."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    x_positions = np.arange(len(MODEL_ORDER))
    overall_lookup = overall.set_index("model")

    for x, model_name in zip(x_positions, MODEL_ORDER):
        fold_values = by_fold.loc[by_fold["model"] == model_name, "mae"].to_numpy()
        overall_mae = float(overall_lookup.loc[model_name, "mae"])
        color = MODEL_COLORS[model_name]
        ax.bar(x, overall_mae, width=0.58, color=color, alpha=0.78)
        offsets = np.linspace(-0.12, 0.12, len(fold_values))
        ax.scatter(np.full(len(fold_values), x) + offsets, fold_values, color="#111827", s=34, zorder=3, label="MAE dei fold" if x == 0 else None)
        ax.text(x, overall_mae + 0.15, f"{overall_mae:.2f}", ha="center", va="bottom", fontsize=11)

    ax.set_xticks(x_positions, [MODEL_LABELS[name] for name in MODEL_ORDER])
    ax.set_ylabel("MAE (anni)")
    ax.set_title("Errore assoluto medio complessivo e per fold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    return save_figure(fig, "mae_by_model.png")


def make_site_mae_figure(by_site):
    """Confronta la MAE per i siti con almeno MIN_SITE_SIZE persone."""
    pivot = by_site.pivot(index="site", columns="model", values="mae")
    pivot = pivot.dropna(subset=MODEL_ORDER)
    pivot = pivot.sort_values("ridge", ascending=True)
    y_positions = np.arange(len(pivot))
    offsets = {"dummy_median": -0.22, "ridge": 0.0, "mlp": 0.22}

    fig_height = max(6, 0.42 * len(pivot) + 1.8)
    fig, ax = plt.subplots(figsize=(9, fig_height))
    for model_name in MODEL_ORDER:
        ax.scatter(
            pivot[model_name],
            y_positions + offsets[model_name],
            s=48,
            color=MODEL_COLORS[model_name],
            marker=MODEL_MARKERS[model_name],
            label=MODEL_LABELS[model_name],
        )
    ax.set_yticks(y_positions, pivot.index)
    ax.set_xlabel("MAE (anni)")
    ax.set_ylabel("Sito di acquisizione")
    ax.set_title(f"MAE per sito (almeno {MIN_SITE_SIZE} partecipanti)")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, ncol=3, loc="lower right")
    return save_figure(fig, "mae_by_site.png")


def build_verification_checklist(predictions, cohort, exclusions, folds):
    """Costruisce una checklist verificabile degli output principali."""
    checks = [
        ("cohort_ids_unique", cohort["subject_id"].is_unique, f"n={len(cohort)}"),
        ("all_exclusions_have_reason", exclusions.loc[~exclusions["included"], "exclusion_reason"].fillna("").str.strip().ne("").all(), f"excluded={int((~exclusions['included']).sum())}"),
        ("sites_not_split", folds.groupby("site")["outer_fold"].nunique().eq(1).all(), f"sites={folds['site'].nunique()}"),
        ("one_prediction_per_subject_model", not predictions.duplicated(["model", "subject_id"]).any(), f"rows={len(predictions)}"),
        ("all_models_complete", all(len(predictions.loc[predictions["model"] == model]) == len(cohort) for model in MODEL_ORDER), f"models={len(MODEL_ORDER)}"),
        ("sex_recorded_for_all_participants", cohort["SEX"].notna().all(), f"missing={int(cohort['SEX'].isna().sum())}"),
        ("predictions_finite", np.isfinite(predictions[["age_true", "age_pred", "bag", "absolute_error"]].to_numpy(dtype=float)).all(), "age_true, age_pred, BAG, absolute_error"),
        ("negative_results_preserved", set(predictions["model"]) == set(MODEL_ORDER), "Dummy, Ridge, MLP"),
    ]
    checklist = pd.DataFrame(checks, columns=["check", "passed", "details"])
    if not checklist["passed"].all():
        failed = checklist.loc[~checklist["passed"], "check"].tolist()
        raise RuntimeError(f"Controlli finali falliti: {failed}")
    return checklist


def sha256_file(path):
    """Calcola l'impronta SHA-256 di un file."""
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def freeze_final_outputs(table_paths, figure_paths):
    """Copia configurazione, risultati e figure nella cartella finale."""
    destinations = []
    log_paths = sorted(path for path in LOGS_DIR.iterdir() if path.is_file())
    copy_groups = {
        FINAL_DIR / "tables": table_paths,
        FINAL_DIR / "figures": figure_paths,
        FINAL_DIR / "predictions": [PREDICTIONS_PATH],
        FINAL_DIR / "logs": log_paths,
        FINAL_DIR / "configuration": [
            CONFIG_PATH,
            VERSIONS_PATH,
            REQUIREMENTS_PATH,
            COHORT_PATH,
            EXCLUSIONS_PATH,
            FOLDS_PATH,
            FOLD_SUMMARY_PATH,
        ],
    }

    for destination_dir, source_paths in copy_groups.items():
        destination_dir.mkdir(parents=True, exist_ok=True)
        for source in source_paths:
            if not source.exists():
                continue
            destination = destination_dir / source.name
            shutil.copy2(source, destination)
            destinations.append(destination)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen",
        "files": [
            {
                "path": str(path.relative_to(FINAL_DIR)),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(destinations)
        ],
    }
    manifest_path = FINAL_DIR / "manifest.json"
    save_json_atomically(manifest, manifest_path)
    return manifest_path


def main():
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    predictions, cohort, exclusions, folds = load_and_validate_inputs()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    overall, by_fold, by_site = compute_metric_tables(predictions)
    bag_analysis, movement_analysis = compute_bag_and_confound_tables(predictions)
    age_band_metrics = compute_age_band_metrics(predictions)
    cohort_description = compute_cohort_description(cohort)
    sex_metrics = compute_sex_metrics(predictions)
    checklist = build_verification_checklist(predictions, cohort, exclusions, folds)

    table_data = {
        "metrics_overall.csv": overall,
        "metrics_by_fold.csv": by_fold,
        "metrics_by_site.csv": by_site,
        "bag_analysis.csv": bag_analysis,
        "movement_error_association.csv": movement_analysis,
        "metrics_by_age_band.csv": age_band_metrics,
        "cohort_description.csv": cohort_description,
        "metrics_by_sex.csv": sex_metrics,
        "verification_checklist.csv": checklist,
    }
    table_paths = []
    for filename, dataframe in table_data.items():
        destination = TABLES_DIR / filename
        save_csv_atomically(dataframe, destination)
        table_paths.append(destination)

    figure_paths = [
        make_cohort_flow_figure(exclusions),
        make_age_distribution_figure(cohort),
        make_actual_vs_predicted_figure(predictions, overall),
        make_bag_figure(predictions, bag_analysis),
        make_model_mae_figure(overall, by_fold),
        make_site_mae_figure(by_site),
    ]

    # Le due funzioni grafiche precedenti producono anche le proprie tabelle sorgente.
    table_paths.extend(
        [TABLES_DIR / "cohort_flow.csv", TABLES_DIR / "age_histogram.csv"]
    )

    report_log = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "minimum_site_size": MIN_SITE_SIZE,
        "models": MODEL_ORDER,
        "n_participants": len(cohort),
        "n_predictions": len(predictions),
        "tables": [path.name for path in table_paths],
        "figures": [path.name for path in figure_paths],
    }
    save_json_atomically(report_log, LOGS_DIR / "report_run.json")
    manifest_path = freeze_final_outputs(table_paths, figure_paths)

    print("Metriche complessive:")
    print(overall.to_string(index=False))
    print("\nAnalisi BAG e calibrazione:")
    print(bag_analysis.to_string(index=False))
    print("\nDescrizione della coorte:")
    print(cohort_description.to_string(index=False))
    print("\nMetriche descrittive per codice di sesso:")
    print(sex_metrics.to_string(index=False))
    print("\nControlli finali:")
    print(checklist.to_string(index=False))
    print("\nOutput creati:")
    print(f"Tabelle: {len(table_paths)} in {TABLES_DIR}")
    print(f"Figure: {len(figure_paths)} in {FIGURES_DIR}")
    print("Manifest finale:", manifest_path)


if __name__ == "__main__":
    main()
