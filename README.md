# despite_fraud_API

Servizio REST API realizzato con **FastAPI** per l'inferenza su dati spettroscopici NIR (spettrometria nel vicino infrarosso) per identificare frodi alimentari (pesce fresco vs decongelato).

Il servizio carica i modelli in modalità **lazy loading** con thread-safety (utilizzando il double-checked locking) ed esegue il preprocessing dei dati spettroscopici (filtro Savitzky-Golay e normalizzazione SNV) prima dell'inferenza.

---

## Prerequisiti

* Python 3.12 (o compatibile)
* Il modello pre-addestrato del polpo deve essere presente nella directory adiacente al percorso:
  `../octopus_nir_classification/model/cnn/cnn_ensemble.keras`

---

## Configurazione del Progetto

1. Spostati nella cartella del progetto:
   ```bash
   cd c:\Users\peiretti\Documents\despite_fraud_API
   ```

2. Crea l'ambiente virtuale:
   ```bash
   python -m venv venv
   ```

3. Attiva l'ambiente virtuale:
     ```cmd
     .\venv\Scripts\activate
     ```

4. Installa le dipendenze:
   ```bash
   pip install -r requirements.txt
   ```

---

## Avvio del Server

Avvia il server di sviluppo tramite **Uvicorn**:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Una volta avviato, la documentazione interattiva OpenAPI (Swagger UI) sarà disponibile all'indirizzo:
* [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## Esecuzione dei Test

Per eseguire i test automatici di validazione, lazy loading e predizione sul polpo (con pytest):

```bash
pytest tests/test_api.py -v
```
