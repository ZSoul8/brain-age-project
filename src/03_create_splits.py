import json #per leggere run_config.json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold #realizza la divisione per sito


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features_cc200.npz"
COHORT_PATH = PROJECT_ROOT / "data" / "interim" / "cohort.csv"
CONFIG_PATH = PROJECT_ROOT / "run_config.json"
FOLDS_PATH = PROJECT_ROOT / "data" / "interim" / "outer_folds.csv"
SUMMARY_PATH = PROJECT_ROOT / "data" / "interim" / "outer_fold_summary.csv"


def create_grouped_folds(y, groups, n_splits): #riceve età dei 458 partecipanti, sito di ciascun partecipante e numero di fold (5)
    """Assegna ogni partecipante a un fold senza dividere i siti."""
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups, dtype=str)

    if y.ndim != 1 or groups.ndim != 1: #entrambi devono essere vettori
        raise ValueError("y e groups devono essere vettori monodimensionali")

    if len(y) != len(groups): #A ogni età deve corrispondere esattamente un sito
        raise ValueError("y e groups devono avere la stessa lunghezza")

    unique_sites = np.unique(groups) #trova il numero di siti distinti (in questo caso 20)
    if len(unique_sites) < n_splits:
        raise ValueError(
            f"Servono almeno {n_splits} siti, trovati {len(unique_sites)}"
        )

    outer_cv = GroupKFold(n_splits=n_splits) #crea il divisore in 5 fold, tratta il dito come un'unità indivisibile
    fold_assignment = np.full(len(y), fill_value=-1, dtype=np.int32) #prepara un vettore di 458 posizioni con -1
    placeholder_X = np.zeros((len(y), 1), dtype=np.uint8) #l'interfaccia GroupKFold richiede un argomento X, ma la divisione dipende solo da groups, non è necessario caricare la matrice reale, viene usata una matrice fittizia

    for fold, (train_idx, test_idx) in enumerate( #cucka su ogni fold e restituisce posizioni dei partecipanti di training e di test
        outer_cv.split(placeholder_X, y, groups=groups)
    ):
        #crea gli insiemi dei siti presenti nel training e nel test
        train_sites = set(groups[train_idx])
        test_sites = set(groups[test_idx])

        if not train_sites.isdisjoint(test_sites): #controllo anti-leakage
            raise RuntimeError(f"Leakage di sito rilevato nel fold {fold}")

        if not set(train_idx).isdisjoint(set(test_idx)):
            raise RuntimeError(f"Partecipanti duplicati nel fold {fold}")

        if np.any(fold_assignment[test_idx] != -1):
            raise RuntimeError("Un partecipante è stato assegnato più volte")

        fold_assignment[test_idx] = fold #registra il fold nel quale ciascun partecipante sarà usato come test

    validate_fold_assignment(groups, fold_assignment, n_splits) #richiama un secondo controllo indipendente
    return fold_assignment


def validate_fold_assignment(groups, fold_assignment, n_splits):
    """Controlla copertura, valori dei fold e separazione completa dei siti."""
    groups = np.asarray(groups, dtype=str)
    fold_assignment = np.asarray(fold_assignment)

    if len(groups) != len(fold_assignment):
        raise ValueError("groups e fold_assignment hanno lunghezze diverse")

    expected_folds = set(range(n_splits)) #con 5 fold produce {0,1,2,3,4}
    observed_folds = set(np.unique(fold_assignment).tolist()) #recupera i fold effettivamente presenti, se manca un fold oppure è rimasto un -1 il programma di ferma
    if observed_folds != expected_folds:
        raise ValueError(
            f"Fold osservati {sorted(observed_folds)}, "
            f"attesi {sorted(expected_folds)}"
        )

    for site in np.unique(groups): #per ogni sito recupera tutti i fold, il risultato deve ottenere un solo valore per essere corretto
        site_folds = np.unique(fold_assignment[groups == site])
        if len(site_folds) != 1:
            raise ValueError(
                f"Il sito {site} è stato diviso tra i fold {site_folds.tolist()}"
            )


def build_fold_summary(folds, n_splits):
    """Riassume numerosità, siti e distribuzione dell'età di ogni test fold."""
    records = []

    for fold in range(n_splits):
        test_rows = folds.loc[folds["outer_fold"] == fold] #prendi tutti i soggetti assegnati come test al fold corrente
        train_rows = folds.loc[folds["outer_fold"] != fold] #prendi tutti i soggetti non assegnati come test al fold corrente
        test_sites = sorted(test_rows["site"].unique().tolist()) #prende i siti presenti nei test
        train_sites = sorted(train_rows["site"].unique().tolist()) #prende i siti presenti nel train

        if not set(train_sites).isdisjoint(test_sites):
            raise RuntimeError(f"Leakage di sito nell'audit del fold {fold}")

        records.append(
            {
                "outer_fold": fold,
                "n_train": len(train_rows),
                "n_test": len(test_rows),
                "n_train_sites": len(train_sites),
                "n_test_sites": len(test_sites),
                "test_sites": ";".join(test_sites),
                "test_age_min": test_rows["age"].min(),
                "test_age_max": test_rows["age"].max(),
                "test_age_mean": test_rows["age"].mean(),
                "test_age_std": test_rows["age"].std(ddof=1),
            }
        )

    return pd.DataFrame(records)


def load_inputs():
    """Carica configurazione, feature e coorte e ne verifica l'allineamento."""
    for path in (FEATURES_PATH, COHORT_PATH, CONFIG_PATH): #controllo esistenza dei path
        if not path.exists():
            raise FileNotFoundError(f"File necessario non trovato: {path}")

    with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    n_splits = int(config["outer_splits"])
    if n_splits < 2:
        raise ValueError("outer_splits deve essere almeno 2")

    with np.load(FEATURES_PATH, allow_pickle=False) as features: #lettura delle feature (apre features_cc200.npz)
        #non viene caricata X, dato che per gli split non servono le 19.900 connessioni
        y = features["y"].copy()
        groups = features["groups"].astype(str)
        subject_ids = features["subject_ids"].astype(str)
        source_indices = features["source_indices"].copy()

    #controlla che tutti gli oggetti abbiano la stessa lunghezza
    cohort = pd.read_csv(COHORT_PATH)

    lengths = {
        len(y),
        len(groups),
        len(subject_ids),
        len(source_indices),
        len(cohort),
    }
    if len(lengths) != 1: #se tutte le lunghezze sono 458, l'insieme contiene solo un elemento
        raise RuntimeError("Feature e coorte hanno lunghezze diverse")

    #controlli di corrispondenza, ID, siti, età, source_index, ID unici, età finite
    if not np.isfinite(y).all():
        raise ValueError("y contiene valori non finiti")

    if len(np.unique(subject_ids)) != len(subject_ids):
        raise ValueError("Gli identificativi dei partecipanti non sono unici")

    if not np.array_equal(
        subject_ids,
        cohort["subject_id"].astype(str).to_numpy(),
    ):
        raise RuntimeError("Gli ID delle feature non coincidono con cohort.csv")

    if not np.array_equal(
        groups,
        cohort["SITE_ID"].astype(str).to_numpy(),
    ):
        raise RuntimeError("I siti delle feature non coincidono con cohort.csv")

    if not np.allclose( #viene usato allclose perché y usa float32 e possono esserci piccole differenze
        y,
        cohort["AGE_AT_SCAN"].to_numpy(dtype=float),
    ):
        raise RuntimeError("Le età delle feature non coincidono con cohort.csv")

    if not np.array_equal(
        source_indices,
        cohort["source_index"].to_numpy(dtype=np.int32),
    ):
        raise RuntimeError(
            "I source_index delle feature non coincidono con cohort.csv"
        )

    return n_splits, y, groups, subject_ids, source_indices


def save_csv_atomically(dataframe, destination):
    """Salva un CSV completo prima di sostituire l'eventuale file precedente."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(".tmp.csv") #salva prima in un file temporaneo
    dataframe.to_csv(temporary_path, index=False)
    temporary_path.replace(destination) #solo dopo il completamento sostituisce il file definitivo per non avere un CSV incompleto


def main():
    n_splits, y, groups, subject_ids, source_indices = load_inputs() #carica e verifica gli input

    print("Partecipanti:", len(y))
    print("Siti distinti:", len(np.unique(groups)))
    print("Fold esterni richiesti:", n_splits)

    fold_assignment = create_grouped_folds(y, groups, n_splits) #crea l'assegnazione ai 5 fold

    folds = pd.DataFrame( #costruisce la tabella principale
        {
            "subject_id": subject_ids,
            "site": groups,
            "age": y,
            "source_index": source_indices,
            "outer_fold": fold_assignment,
        }
    )
    summary = build_fold_summary(folds, n_splits) #costruisce una tabella riassuntiva

    #salva entrambi i file
    save_csv_atomically(folds, FOLDS_PATH)
    save_csv_atomically(summary, SUMMARY_PATH)

    # Ricarica i file per verificare che il contenuto persistente sia valido.
    saved_folds = pd.read_csv(FOLDS_PATH)
    saved_summary = pd.read_csv(SUMMARY_PATH)
    validate_fold_assignment(
        saved_folds["site"].to_numpy(),
        saved_folds["outer_fold"].to_numpy(),
        n_splits,
    )

    if len(saved_folds) != len(y) or len(saved_summary) != n_splits:
        raise RuntimeError("I file degli split hanno dimensioni inattese")

    print("\nAudit dei test fold:")
    print(summary.to_string(index=False))
    print("\nFile salvati e verificati:")
    print(FOLDS_PATH)
    print(SUMMARY_PATH)


if __name__ == "__main__":
    main()
