# despite_fraud_API

Servizio REST API realizzato con **FastAPI** per l'inferenza su dati spettroscopici NIR (spettrometria nel vicino infrarosso) per identificare frodi alimentari (pesce fresco vs decongelato).

Il servizio carica i modelli in modalità **lazy loading** con thread-safety (utilizzando il double-checked locking) ed esegue il preprocessing dei dati spettroscopici (filtro Savitzky-Golay e normalizzazione SNV) prima dell'inferenza.

---

## Prerequisiti

* Python 3.12 (o compatibile)
* Scaricare la repo Git `octopus_nir_classification` (https://github.com/federicopeiretti/octopus_nir_classification)
* Il modello pre-addestrato del polpo deve essere presente nella directory adiacente al percorso:
  `../octopus_nir_classification/model/cnn/cnn_ensemble.keras`

---

## Configurazione del Progetto

1. Spostati nella cartella del progetto:
   ```bash
   cd ../despite_fraud_API
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

---

## Esempio

Di seguito viene riportato un esempio di richiesta HTTP POST all'endpoint `/predict` e della risposta restituita dal server.

### Esempio di richiesta (cURL)

I vettori `wavelength` e `absorbance` devono avere 125 elementi per lo strumento `nir`.

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "measurement_type": "nir",
    "tipo di campione": "pesce",
    "specie campione": "polpo",
    "wavelength": [900.0, 906.45, 912.9, "... (125 elementi in totale)"],
    "absorbance": [0.3745, 0.9507, 0.7319, "... (125 elementi in totale)"]
  }'
```

### Esempio di richiesta (Python)

Ecco come generare una richiesta valida programmando in Python usando la libreria `requests`:

```python
import requests
import numpy as np

# Genera 125 valori fittizi per i vettori di lunghezza d'onda e assorbanza
wavelength = np.linspace(900, 1700, 125).tolist()
absorbance = np.random.rand(125).tolist()

payload = {
    "measurement_type": "nir",
    "tipo di campione": "pesce",
    "specie campione": "polpo",
    "wavelength": wavelength,
    "absorbance": absorbance
}

response = requests.post("http://127.0.0.1:8000/predict", json=payload)
print(response.json())
```

### Risposta (JSON)

In caso di successo (HTTP 200), il server risponde con la classificazione del campione e la probabilità associata:

```json
{
  "label": "decongelato",
  "probability": 0.8421832084655762
}
```

