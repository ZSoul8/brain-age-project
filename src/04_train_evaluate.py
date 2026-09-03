import argparse #permette di scegliere modelli e fold da terminale
import json
import time #per la durata del training
import warnings
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory #crea una cache temporanea per accelerare la cross-validation

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA #riduce le 19.900 connessioni a 64 componenti per la MLP
from sklearn.dummy import DummyRegressor #predice sempre la mediana del training
from sklearn.feature_selection import VarianceThreshold #elimina eventuali feature senza variazione
from sklearn.linear_model import Ridge #modello lineare regolarizzato
from sklearn.model_selection import GridSearchCV, GroupKFold #divide i dati mantenendo separati i siti e sceglie gli iperparametri nei fold interni
from sklearn.neural_network import MLPRegressor #rete neurale multistrato
from sklearn.pipeline import Pipeline #mantiene trasformazioni e modello in una sequenza indivisibile
from sklearn.preprocessing import StandardScaler #centra e standardizza


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features_cc200.npz"
FOLDS_PATH = PROJECT_ROOT / "data" / "interim" / "outer_folds.csv"
COHORT_PATH = PROJECT_ROOT / "data" / "interim" / "cohort.csv"
CONFIG_PATH = PROJECT_ROOT / "run_config.json"
PREDICTIONS_PATH = (
    PROJECT_ROOT / "results" / "predictions" / "predictions.csv"
)
LOGS_DIR = PROJECT_ROOT / "results" / "logs"

AVAILABLE_MODELS = ("dummy_median", "ridge", "mlp")
PREDICTION_COLUMNS = [ #stabilisce le colonne finali
    "subject_id",
    "site",
    "outer_fold",
    "model",
    "age_true",
    "age_pred",
    "bag",
    "absolute_error",
    "source_index",
    "sex",
    "func_mean_fd",
    "func_num_fd",
    "func_perc_fd",
]

#funzioni per il salvataggio sicuro attraverso uno manuale
def save_csv_atomically(dataframe, destination):
    """Salva un CSV completo prima di sostituire il file precedente."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(".tmp.csv")
    dataframe.to_csv(temporary_path, index=False)
    temporary_path.replace(destination)


def save_json_atomically(data, destination):
    """Salva un log JSON completo prima di sostituire il precedente."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(".tmp.json")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=2, ensure_ascii=False)
    temporary_path.replace(destination)


def load_inputs():
    """Carica e allinea feature, fold, coorte e configurazione."""
    for path in (FEATURES_PATH, FOLDS_PATH, COHORT_PATH, CONFIG_PATH): #controlla se i file esistano
        if not path.exists():
            raise FileNotFoundError(f"File necessario non trovato: {path}")

    with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    #apre features_cc200.npz e carica le seguenti cose
    with np.load(FEATURES_PATH, allow_pickle=False) as features:
        X = features["X"].copy() #X.shape = (458, 19900)
        y = features["y"].astype(np.float64) #y.shape = (458,)
        groups = features["groups"].astype(str)
        subject_ids = features["subject_ids"].astype(str)
        source_indices = features["source_indices"].astype(np.int32)

    #caricamento
    folds = pd.read_csv(FOLDS_PATH)
    cohort = pd.read_csv(COHORT_PATH)

    lengths = {
        len(X),
        len(y),
        len(groups),
        len(subject_ids),
        len(source_indices),
        len(folds),
        len(cohort),
    }
    #controlli
    if len(lengths) != 1:
        raise RuntimeError("Feature, fold e coorte hanno lunghezze diverse")

    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X deve essere 2D e y deve essere 1D")

    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("X o y contengono valori non finiti")

    if not np.array_equal(subject_ids, folds["subject_id"].astype(str)):
        raise RuntimeError("Gli ID delle feature non coincidono con i fold")

    if not np.array_equal(groups, folds["site"].astype(str)):
        raise RuntimeError("I siti delle feature non coincidono con i fold")

    if not np.allclose(y, folds["age"].to_numpy(dtype=float)):
        raise RuntimeError("Le età delle feature non coincidono con i fold")

    if not np.array_equal(
        source_indices,
        folds["source_index"].to_numpy(dtype=np.int32),
    ):
        raise RuntimeError("I source_index non coincidono con i fold")

    if not np.array_equal(subject_ids, cohort["subject_id"].astype(str)):
        raise RuntimeError("Gli ID delle feature non coincidono con la coorte")

    return X, y, groups, subject_ids, source_indices, folds, cohort, config


def build_ridge_pipeline(cache_dir):
    """Crea la pipeline Ridge e la griglia fissata nel protocollo."""
    pipeline = Pipeline(
        [
            ("variance", VarianceThreshold()), #elimina eventuali connessioni che non variano nel training corrente
            ("scaler", StandardScaler()), #per ogni feature calcola media e deviazione standard usando solo il training
            ("model", Ridge()), #impara una combinazione lineare delle connessioni per prevedere l’età
        ],
        memory=cache_dir,
    )
    param_grid = {
        "model__alpha": [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000] #non sapendo quale quantità dia migliore le provo e scelgo quella che funziona meglio
    }
    return pipeline, param_grid


def build_mlp_pipeline(cache_dir, seed):
    """Crea la pipeline MLP e la piccola griglia fissata nel protocollo."""
    pipeline = Pipeline(
        [
            ("variance", VarianceThreshold()),
            ("scaler", StandardScaler()),
            (
                "pca",
                PCA( #riduce le 19.900 connessioni a 64 componenti (riducendo la probabilità di overfitting)
                    n_components=64,
                    svd_solver="randomized",
                    random_state=seed,
                ),
            ),
            (
                "model",
                MLPRegressor(
                    hidden_layer_sizes=(64, 32), #sono presenti due hidden layer con questo numero di neuroni
                    activation="relu",
                    solver="adam", #aggiorna i pesi durante il training
                    #il 15% del training corrente viene usato per osservare se il modello continua a migliorare
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=20, #interrompe l'addestramento se per 20 epoche non c'è miglioramento sufficiente
                    max_iter=500, #inizialmente impostata a 300 ma alcune configurazioni non convergevano
                    random_state=seed,
                ),
            ),
        ],
        memory=cache_dir,
    )
    param_grid = { #vengono provate queste quattro combinazioni
        "model__alpha": [1e-4, 1e-3],
        "model__learning_rate_init": [1e-4, 1e-3],
    }
    return pipeline, param_grid


def check_inner_cv_capacity(X_train, y_train, groups_train, inner_splits):
    """Verifica siti e dimensione minima dei training fold interni."""
    unique_train_sites = np.unique(groups_train) #trova i siti distinti nel training esterno
    #se ci sono meno siti del numero di fold interni richiesto non può essere fatta la divisione
    if len(unique_train_sites) < inner_splits:
        raise ValueError(
            f"Servono almeno {inner_splits} siti nel training esterno, "
            f"trovati {len(unique_train_sites)}"
        )

    inner_cv = GroupKFold(n_splits=inner_splits) #crea il divisore e soggetti appartenenti allo stesso sito non devono essere separati
    #lo facciamo per vedere se il modello riesce a generalizzare soggetti provenienti da siti non visti nell'addestramento
    inner_train_sizes = [ #simulo una cross-validation e raccolgo le dimensioni per fold interno
        len(inner_train_idx)
        for inner_train_idx, _ in inner_cv.split(
            X_train,
            y_train,
            groups=groups_train,
        )
    ]
    minimum_inner_train = min(inner_train_sizes) #trova il training più piccolo

    if minimum_inner_train <= 64: #per 64 componenti PCA devo avere necessariamente almeno 64 soggetti nel training
        raise ValueError(
            "Il più piccolo training fold interno contiene "
            f"{minimum_inner_train} persone: PCA(64) non è applicabile"
        )

    return minimum_inner_train #verrà registrata nel log


def run_dummy(X_train, y_train, X_test):
    """Addestra la baseline che predice la mediana del training."""
    model = DummyRegressor(strategy="median") #crea un regressore che usa la mediana di y_train 
    model.fit(X_train, y_train) #calcola e memorizza la mediana
    predictions = model.predict(X_test) #produce una predizione per ogni riga di X_test

    if not np.allclose(predictions, predictions[0]): #controlla che le predizioni siano uguali a prima (deve essere così nel dummy)
        raise RuntimeError("DummyRegressor ha prodotto valori non costanti")

    return predictions, {"strategy": "median"}, None, []


def run_grid_model(
    model_name,
    X_train,
    y_train,
    groups_train,
    X_test,
    inner_splits,
    seed,
):
    """Esegue la ricerca interna per Ridge o MLP e predice il test esterno."""
    minimum_inner_train = check_inner_cv_capacity( #verificano la capacità della CV interna e conservano la dimensione minima
        X_train,
        y_train,
        groups_train,
        inner_splits,
    )

    inner_cv = GroupKFold(n_splits=inner_splits) #crea un GroupKFold interno

    with TemporaryDirectory(prefix=f"brain_age_{model_name}_") as cache_dir: #cache temporanea
        #costruisce le diverse pipeline a seconda del modello
        if model_name == "ridge":
            pipeline, param_grid = build_ridge_pipeline(cache_dir)
        elif model_name == "mlp":
            pipeline, param_grid = build_mlp_pipeline(cache_dir, seed)
        else:
            raise ValueError(f"Modello non supportato: {model_name}")

        search = GridSearchCV( #serve a trovare i migliori iperparametri, provo tutte le configurazioni e scelgo quella con MAE più basso
            estimator=pipeline, #stimatore da ottimizzare
            param_grid=param_grid, #griglia degli iperparametri
            scoring="neg_mean_absolute_error", #Scikit-learn massimizza gli score --> il valore migliore è quello più vicino allo zero
            cv=inner_cv,
            n_jobs=1,
            refit=True, #dopo aver scelto gli iperparametri migliori, riaddestra la pipeline su tutto il training esterno
            return_train_score=False,
            error_score="raise",
        )

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always") 
            search.fit(X_train, y_train, groups=groups_train) #esegue la ricerca
            predictions = search.predict(X_test) #usa i parametri migliori per predire il test esterno

        warning_messages = sorted(
            {str(item.message) for item in caught_warnings}
        )
        #inizializza la lista dei risultati dei candidati
        candidate_results = []
        for params, mean_score, std_score in zip(
            search.cv_results_["params"],
            search.cv_results_["mean_test_score"],
            search.cv_results_["std_test_score"],
        ):
            candidate_results.append( #aggiungo un dizionario per ogni configurazione
                {
                    "params": params,
                    "mean_inner_mae": float(-mean_score), #cambio segno allo score per riportare il MAE positivo
                    "std_inner_mae": float(std_score),
                }
            )

        best_params = {
            key: value.item() if isinstance(value, np.generic) else value
            for key, value in search.best_params_.items()
        }
        search_details = {
            "minimum_inner_train_size": minimum_inner_train,
            "best_inner_mae": float(-search.best_score_),
            "candidates": candidate_results,
        }

    return predictions, best_params, search_details, warning_messages


def build_prediction_rows(
    test_idx,
    fold,
    model_name,
    y,
    y_pred,
    groups,
    subject_ids,
    source_indices,
    cohort,
):
    """Costruisce le righe descrittive delle predizioni esterne."""
    y_test = y[test_idx] #estrae le età reali
    y_pred = np.asarray(y_pred, dtype=float) #conversione in un array numerico

    if y_pred.shape != y_test.shape or not np.isfinite(y_pred).all():
        raise RuntimeError("Le predizioni hanno forma o valori non validi")

    descriptive = cohort.iloc[test_idx] #estrae dalla coorte le righe corrispondenti agli indici di test
    return pd.DataFrame( #costruisco il dataframe finale
        {
            "subject_id": subject_ids[test_idx],
            "site": groups[test_idx],
            "outer_fold": fold,
            "model": model_name,
            "age_true": y_test,
            "age_pred": y_pred,
            "bag": y_pred - y_test, #calcolo BAG
            "absolute_error": np.abs(y_pred - y_test),
            "source_index": source_indices[test_idx],
            "sex": descriptive["SEX"].to_numpy(),
            "func_mean_fd": descriptive["func_mean_fd"].to_numpy(),
            "func_num_fd": descriptive["func_num_fd"].to_numpy(),
            "func_perc_fd": descriptive["func_perc_fd"].to_numpy(),
        }
    )


def load_existing_predictions():
    """Ricarica eventuali risultati parziali per consentire la ripresa."""
    if not PREDICTIONS_PATH.exists(): #se il CSV non esiste, restituisce un dataframe vuoto con le colonne corrette
        return pd.DataFrame(columns=PREDICTION_COLUMNS)

    predictions = pd.read_csv(PREDICTIONS_PATH) #legge predizioni già presenti
    missing_columns = set(PREDICTION_COLUMNS) - set(predictions.columns)
    if missing_columns:
        raise RuntimeError(
            f"predictions.csv non contiene le colonne {sorted(missing_columns)}"
        )
    return predictions[PREDICTION_COLUMNS]


def combination_is_complete(predictions, model_name, fold, expected_ids):
    """Controlla se una combinazione modello-fold è già stata completata."""
    rows = predictions.loc[ #selezionano dal CSV soltanto le righe appartenenti al modello e al fold richiesti
        (predictions["model"] == model_name)
        & (predictions["outer_fold"] == fold)
    ]
    return ( #la combinazione è completa solo se tutte le condizioni sono vere
        len(rows) == len(expected_ids)
        and rows["subject_id"].is_unique
        and set(rows["subject_id"].astype(str)) == set(expected_ids)
        and np.isfinite(rows["age_pred"].to_numpy(dtype=float)).all()
    )


def validate_requested_predictions(
    predictions,
    models,
    selected_folds,
    fold_assignment,
    subject_ids,
):
    """Verifica completezza e unicità delle predizioni richieste."""
    #iterano su ogni modello e fold richiesto
    for model_name in models:
        for fold in selected_folds:
            expected_ids = subject_ids[fold_assignment == fold].astype(str) #determina quali soggetti devono appartenere a quel fold
            if not combination_is_complete( #controllo di completezza
                predictions,
                model_name,
                fold,
                expected_ids,
            ):
                raise RuntimeError(
                    f"Predizioni incomplete per {model_name}, fold {fold}"
                )


def main():
    #argomenti da terminale
    parser = argparse.ArgumentParser(
        description="Addestra Dummy, Ridge e MLP con validazione annidata."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=AVAILABLE_MODELS,
        help="Modelli da eseguire; per default usa run_config.json.",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        help="Fold esterni da eseguire; per default li esegue tutti.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ricalcola anche le combinazioni modello-fold già complete.",
    )
    args = parser.parse_args()

    (
        X,
        y,
        groups,
        subject_ids,
        source_indices,
        folds,
        cohort,
        config,
    ) = load_inputs()

    outer_splits = int(config["outer_splits"])
    inner_splits = int(config["inner_splits"])
    seed = int(config["seed"])
    configured_models = list(config["models"])
    models = args.models if args.models is not None else configured_models
    selected_folds = (
        args.folds if args.folds is not None else list(range(outer_splits))
    )

    if not set(models).issubset(AVAILABLE_MODELS):
        raise ValueError(f"Modelli non validi: {models}")

    if not set(selected_folds).issubset(set(range(outer_splits))):
        raise ValueError(f"Fold non validi: {selected_folds}")

    fold_assignment = folds["outer_fold"].to_numpy(dtype=int)
    predictions = load_existing_predictions()

    print("Forma X:", X.shape)
    print("Partecipanti:", len(y))
    print("Fold esterni:", outer_splits)
    print("Fold interni:", inner_splits)
    print("Modelli richiesti:", ", ".join(models))

    for model_name in models:
        for fold in selected_folds:
            train_idx = np.flatnonzero(fold_assignment != fold)
            test_idx = np.flatnonzero(fold_assignment == fold)
            expected_ids = subject_ids[test_idx].astype(str)

            if (
                not args.force
                and combination_is_complete(
                    predictions,
                    model_name,
                    fold,
                    expected_ids,
                )
            ):
                print(f"Saltato {model_name}, fold {fold}: già completo.")
                continue

            train_sites = set(groups[train_idx])
            test_sites = set(groups[test_idx])
            if not train_sites.isdisjoint(test_sites):
                raise RuntimeError(f"Leakage di sito nel fold esterno {fold}")

            X_train = X[train_idx]
            y_train = y[train_idx]
            groups_train = groups[train_idx]
            X_test = X[test_idx]

            print(
                f"\nAvvio {model_name}, fold {fold}: "
                f"train={len(train_idx)}, test={len(test_idx)}",
                flush=True,
            )
            start = time.perf_counter()

            if model_name == "dummy_median":
                y_pred, best_params, search_details, warning_messages = run_dummy(
                    X_train,
                    y_train,
                    X_test,
                )
            else:
                (
                    y_pred,
                    best_params,
                    search_details,
                    warning_messages,
                ) = run_grid_model(
                    model_name,
                    X_train,
                    y_train,
                    groups_train,
                    X_test,
                    inner_splits,
                    seed,
                )

            elapsed_seconds = time.perf_counter() - start
            new_rows = build_prediction_rows(
                test_idx,
                fold,
                model_name,
                y,
                y_pred,
                groups,
                subject_ids,
                source_indices,
                cohort,
            )

            same_combination = (
                (predictions["model"] == model_name)
                & (predictions["outer_fold"] == fold)
            )
            predictions = predictions.loc[~same_combination]
            predictions = pd.concat([predictions, new_rows], ignore_index=True)
            predictions = predictions.sort_values(
                ["model", "outer_fold", "subject_id"]
            ).reset_index(drop=True)
            save_csv_atomically(predictions, PREDICTIONS_PATH)

            log_data = {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "model": model_name,
                "outer_fold": fold,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "train_sites": sorted(train_sites),
                "test_sites": sorted(test_sites),
                "inner_splits": inner_splits if model_name != "dummy_median" else None,
                "best_params": best_params,
                "search": search_details,
                "elapsed_seconds": elapsed_seconds,
                "warnings": warning_messages,
            }
            save_json_atomically(
                log_data,
                LOGS_DIR / f"{model_name}_fold_{fold}.json",
            )

            fold_mae = float(np.mean(new_rows["absolute_error"]))
            print("Parametri:", best_params)
            print(f"MAE esterna del fold: {fold_mae:.3f} anni")
            print(f"Tempo: {elapsed_seconds:.1f} secondi")
            if warning_messages:
                print(f"Avvisi registrati: {len(warning_messages)}")

    validate_requested_predictions(
        predictions,
        models,
        selected_folds,
        fold_assignment,
        subject_ids,
    )
    save_csv_atomically(predictions, PREDICTIONS_PATH)

    requested = predictions.loc[
        predictions["model"].isin(models)
        & predictions["outer_fold"].isin(selected_folds)
    ]
    print("\nPredizioni richieste completate e verificate.")
    print(
        requested.groupby("model")["absolute_error"]
        .agg(["count", "mean"])
        .rename(columns={"mean": "MAE_provvisoria"})
        .to_string()
    )
    print("File:", PREDICTIONS_PATH)


if __name__ == "__main__":
    main()
