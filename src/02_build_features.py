#preparazione del dataset che andrà fornito ai modelli per l'addestramento
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn.datasets import fetch_abide_pcp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path.home() / "brain_age_project_data" / "abide_pcp"
COHORT_PATH = PROJECT_ROOT / "data" / "interim" / "cohort.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "features_cc200.npz"
EXPECTED_ROIS = 200


def timeseries_to_connectome_vector(timeseries, expected_rois=None):
    """Converte una serie tempo × ROI nelle connessioni Pearson Fisher-z."""
    ts = np.asarray(timeseries, dtype=np.float64) #converte l'oggetto in una matrice

    if ts.ndim != 2:
        raise ValueError(f"Attesa matrice 2D, ricevuta forma {ts.shape}")

    n_timepoints, n_rois = ts.shape #instanti della scansione (196), numero di ROI (200)

    if n_timepoints < 2:
        raise ValueError("Servono almeno due campioni temporali") #per calcolare la correlazione

    if expected_rois is not None and n_rois != expected_rois: #controllo viene applicato solo se è stato specificato expected_rois
        raise ValueError(
            f"Numero ROI inatteso: {n_rois}, atteso: {expected_rois}"
        )

    if not np.isfinite(ts).all(): #controlla se ci sono celle con valori finiti
        raise ValueError("La serie contiene NaN o valori infiniti")

    roi_std = np.std(ts, axis=0) #calcola la deviazione standard di ogni colonna
    if np.any(roi_std == 0): #se la deviazione standard è zero vuol dire che è costante e non si può applicare la correlazione di Pearson
        raise ValueError("È presente almeno una ROI costante")

    corr = np.corrcoef(ts, rowvar=False) #calcola la correlazione di Pearson tra ogni coppia di colonne
    #Ogni cella corr[i, j] rappresenta la correlazione tra la ROI i e la ROI j
    #La diagonale contiene sempre 1, perché ogni ROI è perfettamente correlata con sé stessa
    if corr.shape != (n_rois, n_rois):
        raise ValueError("La matrice di correlazione ha forma inattesa")

    if not np.allclose(corr, corr.T, atol=1e-8): #controllo simmetria
        raise ValueError("La matrice di correlazione non è simmetrica")

    upper = np.triu_indices(n_rois, k=1) #prendo solo il triangolo superiore escludendo la diagonale (k=1)
    r_values = corr[upper]

    #trasformazione di Fisher-z
    epsilon = 1e-7
    r_values = np.clip(r_values, -1 + epsilon, 1 - epsilon) #permette di non avere valori infiniti delimitando l'intervallo
    z_values = np.arctanh(r_values)

    if not np.isfinite(z_values).all(): #controllo di valori invalidi
        raise ValueError("Il vettore Fisher-z contiene valori non finiti")

    return z_values.astype(np.float32) #Restituisce le 19.900 feature usando float32


def load_cohort_and_series():
    """Ricarica coorte e serie ABIDE usando la configurazione congelata."""
    if not COHORT_PATH.exists(): #Se cohort.csv non esiste, lo script si ferma e invita a eseguire prima 01_download_abide.py
        raise FileNotFoundError(
            "cohort.csv non trovato. Eseguire prima 01_download_abide.py"
        )

    cohort = pd.read_csv(COHORT_PATH) #carica la coorte definitiva
    abide = fetch_abide_pcp( #fa la stessa configurazione del download
        data_dir=DATA_DIR,
        n_subjects=None,
        pipeline="cpac",
        derivatives=["rois_cc200"],
        quality_checked=True,
        band_pass_filtering=True,
        global_signal_regression=False,
        DX_GROUP=2,
    )

    #separazione tra metadati e serie
    phenotypic = pd.DataFrame(abide.phenotypic).reset_index(drop=True) #468 righe di metadati
    series = list(abide.rois_cc200) #468 matrici cerebrali
    #10 esclusi 

    if len(series) != len(phenotypic): #serie e metadati devono avere la stessa lunghezza
        raise RuntimeError("Serie e metadati ABIDE non sono allineati")

    if cohort.empty:
        raise RuntimeError("La coorte è vuota")

    if int(cohort["source_index"].max()) >= len(series): #ogni source index deve indicare uan posizione esistente dentro series
        raise RuntimeError("Un source_index supera il numero di serie disponibili")

    return cohort, phenotypic, series

#inpedisce di associare l'età di una persona alla serie cerebrale di un'altra 
def get_aligned_timeseries(row, phenotypic, series):
    """Recupera una serie e verifica che appartenga al partecipante atteso."""
    source_index = int(row["source_index"]) #recupera la posizione originale della persona
    expected_file_id = str(row["FILE_ID"])
    actual_file_id = str(phenotypic.iloc[source_index]["FILE_ID"])
    #confronta l'identificativo atteso della coorte e l'identificativo trovato nella posizione originale
    if actual_file_id != expected_file_id:
        raise RuntimeError(
            "Allineamento errato per source_index "
            f"{source_index}: atteso {expected_file_id}, trovato {actual_file_id}"
        )
    #dopo avere escluso 10 persone, il normale indice di cohort.csv non corrisponde più necessariamente alla posizione dentro series
    return series[source_index] #restituzione della serie corretta


def run_pilot(cohort, phenotypic, series):
    """Costruisce e controlla il connectome del primo incluso."""
    first_row = cohort.iloc[0] #prelevo dalla prima riga
    first = get_aligned_timeseries(first_row, phenotypic, series) #recupera la serie verificando l'identificativo
    vector = timeseries_to_connectome_vector( #calcola le connessioni
        first,
        expected_rois=EXPECTED_ROIS,
    )

    expected_features = EXPECTED_ROIS * (EXPECTED_ROIS - 1) // 2 #il risultato è 19.900

    #il pilot deve produrre 19.900 feature e solo valori finiti
    assert vector.shape == (expected_features,)
    assert np.isfinite(vector).all()

    print("Partecipante pilot:", first_row["subject_id"])
    print("Forma serie:", np.asarray(first).shape)
    print("Feature ottenute:", vector.shape[0])
    print("Feature attese:", expected_features)
    print("Tipo numerico:", vector.dtype)
    print("Tutte finite:", bool(np.isfinite(vector).all()))

    return vector

#costruzione della matrice completa, ripete il calcolo per tutti i 458 partecipanti
def build_full_feature_matrix(cohort, phenotypic, series):
    """Costruisce X e i vettori descrittivi per tutti gli inclusi."""
    expected_features = EXPECTED_ROIS * (EXPECTED_ROIS - 1) // 2
    X = np.empty((len(cohort), expected_features), dtype=np.float32) #preparazione della matrice principale X.shape (458, 19900) ogni riga è un partecipante e ogni colonna una connessione
    y = np.empty(len(cohort), dtype=np.float32) #vettore dell'età y.shape = (458,)
    groups = [] #lista dei siti
    subject_ids = [] #lista degli identificativi
    source_indices = np.empty(len(cohort), dtype=np.int32) #conserva le posizioni originali dei partecipanti

    for output_index, (_, row) in enumerate(cohort.iterrows()): #per ogni partecipante
        timeseries = get_aligned_timeseries(row, phenotypic, series) #recupera la serie corretta
        X[output_index] = timeseries_to_connectome_vector( #calcola le 29.900 feature e le inserisce nella riga corrispondente di X
            timeseries,
            expected_rois=EXPECTED_ROIS,
        )
        y[output_index] = float(row["AGE_AT_SCAN"]) #salvo l'età reale
        groups.append(str(row["SITE_ID"])) #salva il sito
        subject_ids.append(str(row["subject_id"])) #salva l'identificativo
        source_indices[output_index] = int(row["source_index"]) #salva la posizione originale

        completed = output_index + 1 #indicatore di avanzamento
        if completed == 1 or completed % 25 == 0 or completed == len(cohort):
            print(
                f"Connectome completati: {completed}/{len(cohort)}",
                flush=True,
            )

    groups = np.asarray(groups, dtype=str)
    subject_ids = np.asarray(subject_ids, dtype=str)

    assert X.ndim == 2
    assert y.ndim == 1
    assert len(X) == len(y) == len(groups) == len(subject_ids)
    assert len(np.unique(subject_ids)) == len(subject_ids)
    assert np.isfinite(X).all()
    assert np.isfinite(y).all()

    return X, y, groups, subject_ids, source_indices


def save_and_verify_features(X, y, groups, subject_ids, source_indices):
    """Salva la cache compressa e la ricarica per verificarne l'integrità."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    temporary_path = PROCESSED_DIR / "features_cc200.tmp.npz"

    np.savez_compressed( #crea archivio NumPy compresso
        temporary_path,
        X=X,
        y=y,
        groups=groups,
        subject_ids=subject_ids,
        source_indices=source_indices,
    )
    temporary_path.replace(OUTPUT_PATH)

    with np.load(OUTPUT_PATH, allow_pickle=False) as saved: #riapre il file appena creato
        assert saved["X"].shape == X.shape
        assert saved["y"].shape == y.shape
        assert np.isfinite(saved["X"]).all()
        assert np.isfinite(saved["y"]).all()
        assert np.array_equal(saved["groups"], groups)
        assert np.array_equal(saved["subject_ids"], subject_ids)
        assert np.array_equal(saved["source_indices"], source_indices)

    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print("\nFile verificato:", OUTPUT_PATH)
    print("Forma X:", X.shape)
    print("Forma y:", y.shape)
    print("Numero siti:", len(np.unique(groups)))
    print(f"Dimensione compressa: {size_mb:.2f} MB")


def main():
    parser = argparse.ArgumentParser( 
        description="Costruisce le feature di connettività CC200."
    )
    parser.add_argument( # se eseguo python src\02_build_features.py --pilot-only, controlla solo il primo partecipante
        "--pilot-only", 
        action="store_true",
        help="Controlla soltanto il primo partecipante incluso.",
    )
    args = parser.parse_args()

    cohort, phenotypic, series = load_cohort_and_series() #riceve i tre oggetti restituiti dalla funzione
    run_pilot(cohort, phenotypic, series) #controllo sul primo partecipante

    if args.pilot_only:
        print("Pilot superato: la matrice completa non è stata ancora costruita.")
        return

    X, y, groups, subject_ids, source_indices = build_full_feature_matrix( #elaborazione completa, costruisce tutti gli array
        cohort,
        phenotypic,
        series,
    )
    save_and_verify_features(X, y, groups, subject_ids, source_indices) #li salva e li verifica


if __name__ == "__main__":
    main()
