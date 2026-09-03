# Brain Age da connettività funzionale ABIDE

Progetto di tesi per stimare l'età cronologica di partecipanti di controllo
ABIDE I a partire dalla connettività funzionale resting-state. Le serie
temporali preelaborate con C-PAC e parcellate con l'atlante CC200 vengono
trasformate in 19.900 correlazioni Fisher-z e utilizzate per confrontare una
baseline Dummy, una regressione Ridge e una rete neurale MLP.

## Stato del progetto

La pipeline sperimentale è completa. La coorte finale contiene 458
partecipanti provenienti da 20 siti. I risultati formali sono ottenuti con
validazione annidata e test esterni separati per sito.

| Modello | MAE (anni) | RMSE (anni) | R² | Pearson r |
|---|---:|---:|---:|---:|
| Dummy mediano | 5,13 | 7,56 | -0,10 | -0,23 |
| Ridge | **4,06** | **5,77** | **0,36** | **0,60** |
| MLP | 5,91 | 7,98 | -0,22 | 0,40 |

Ridge è il modello migliore in questo esperimento. Le predizioni mostrano
tuttavia un marcato age bias: le età più giovani tendono a essere
sovrastimate e quelle più alte sottostimate.

## Ambiente

- Windows 11 e PowerShell;
- Python 3.12 (esperimento eseguito con Python 3.12.10);
- ambiente virtuale locale `.venv`;
- dipendenze bloccate in `requirements.txt`;
- seed globale: 42.

Creazione e attivazione dell'ambiente da PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

In Visual Studio Code l'interprete da selezionare è:

```text
brain_age_project\.venv\Scripts\python.exe
```

## Dati

Il download usa `nilearn.datasets.fetch_abide_pcp` con questa configurazione:

- ABIDE I Preprocessed;
- soli controlli (`DX_GROUP=2`);
- pipeline C-PAC;
- derivata `rois_cc200`;
- controllo qualità abilitato;
- filtro passa-banda abilitato;
- global signal regression disabilitata.

Per evitare i problemi di permessi incontrati con OneDrive, la cache completa
non si trova dentro il repository. Gli script la cercano in:

```text
C:\Users\<nome-utente>\brain_age_project_data\abide_pcp
```

Sul computer usato per l'esperimento la cache contiene 468 serie temporali e
occupa circa 172 MB. Se la cache non esiste, Nilearn prova a scaricarla di
nuovo. I dati grezzi, i dati individuali e le predizioni con identificativi
non devono essere pubblicati in un repository GitHub pubblico.

## Struttura principale

```text
brain_age_project/
├── README.md
├── requirements.txt
├── versions.txt
├── run_config.json
├── data/
│   ├── raw/          # cache locale o download parziali, non versionati
│   ├── interim/      # coorte, esclusioni e fold
│   └── processed/    # matrice features_cc200.npz
├── src/
│   ├── 01_download_abide.py
│   ├── 02_build_features.py
│   ├── 03_create_splits.py
│   ├── 04_train_evaluate.py
│   └── 05_make_report.py
├── results/
│   ├── predictions/
│   ├── tables/
│   ├── figures/
│   ├── logs/
│   └── final/
└── tests/
```

## Esecuzione della pipeline

Aprire un terminale nella cartella `brain_age_project`, attivare `.venv` ed
eseguire gli script nell'ordine seguente.

### 1. Download e selezione della coorte

```powershell
python src\01_download_abide.py
```

Produce `cohort.csv`, `exclusions.csv` e i metadati completi. Il download può
richiedere diversi minuti; se la cache è già presente, Nilearn riutilizza i
file locali.

### 2. Costruzione delle feature

Controllo su una sola persona:

```powershell
python src\02_build_features.py --pilot-only
```

Matrice completa:

```powershell
python src\02_build_features.py
```

Produce `data/processed/features_cc200.npz`, contenente:

```text
X:           (458, 19900)
y:           (458,)
groups:      sito di acquisizione
subject_ids: identificativi allineati
```

### 3. Creazione dei fold esterni

```powershell
python src\03_create_splits.py
```

I cinque fold sono creati con `GroupKFold`: uno stesso sito non può comparire
contemporaneamente nel training e nel test esterno.

### 4. Addestramento e predizioni out-of-fold

```powershell
python src\04_train_evaluate.py
```

Lo script può riprendere un'esecuzione interrotta e salta le combinazioni
modello-fold già complete. Per ricalcolare intenzionalmente una combinazione si
può usare `--force`, ma non va fatto sulla versione definitiva senza registrare
un nuovo esperimento.

La selezione degli iperparametri avviene soltanto nei fold interni, anch'essi
raggruppati per sito. `StandardScaler`, selezione della varianza e PCA restano
dentro le pipeline, evitando leakage dal test.

### 5. Tabelle, grafici e congelamento

```powershell
python src\05_make_report.py
```

Produce metriche complessive, per fold, per sito, per fascia d'età e per sesso;
analisi del brain-age gap e del movimento; descrizione della coorte; sei figure
a 300 dpi; checklist e manifest finale con impronte SHA-256.

## Test

```powershell
python -m pytest tests -q
```

I test controllano la trasformazione delle serie in vettori di connettività,
il rifiuto di input non validi, la separazione dei siti e le nuove tabelle del
report.

## Output da utilizzare nella tesi

- `results/final/predictions/predictions.csv`: predizioni esterne;
- `results/final/tables/`: metriche e tabelle descrittive;
- `results/final/figures/`: figure definitive;
- `results/final/logs/`: parametri, tempi, avvisi e decisioni;
- `results/final/manifest.json`: impronte dei file congelati.

Per il capitolo dei risultati utilizzare lo snapshot in `results/final`. Se
vengono modificati coorte, fold, feature o modelli, la nuova esecuzione deve
essere trattata come un esperimento successivo e non sostituire silenziosamente
quello riportato nella tesi.

## Limiti da discutere

- distribuzione d'età sbilanciata, con soltanto 28 partecipanti dai 30 anni in
  poi;
- forte associazione negativa tra brain-age gap ed età;
- numerosità sbilanciata tra i codici di sesso registrati;
- siti con numerosità e distribuzioni d'età differenti;
- un fold esterno composto dal solo sito NYU;
- MLP peggiore della baseline Ridge;
- assenza di una validazione su un secondo dataset indipendente.

Questi punti non invalidano l'esperimento, ma limitano la generalizzabilità
delle conclusioni.

## Riferimenti software e dati

- [Nilearn: `fetch_abide_pcp`](https://nilearn.github.io/stable/modules/generated/nilearn.datasets.fetch_abide_pcp.html)
- ABIDE I / Preprocessed Connectomes Project;
- C-PAC preprocessing pipeline;
- parcellazione funzionale CC200 di Craddock e collaboratori.

Le citazioni bibliografiche complete devono essere riportate nella tesi.
