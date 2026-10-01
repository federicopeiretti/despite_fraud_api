# despite_fraud_API

Servizio REST API realizzato con **FastAPI** per l'inferenza su dati spettroscopici NIR (spettrometria nel vicino infrarosso) per identificare frodi alimentari (pesce fresco vs decongelato).

Il servizio carica i modelli in modalità **lazy loading** con thread-safety (utilizzando il double-checked locking).

Viene caricato il modello corretto in base al tipo di strumento chemiometrico (NIR, Raman), alla classe animale (pesce, carne) e alla specie (es. polpo, moscardino, seppia per il pesce; wurstel per la carne) per eseguire l'inferenza sui nuovi spettri. Prima dell'inferenza, viene eseguito il preprocessing dei dati spettroscopici. Successivamente l'inferenza viene eseguita e viene restituita la label del campione con la probabilità che il campione appartenga alla classe predetta (es. `fresco` o `decongelato` per il pesce; `non_csm` o `csm` per la carne).


---

## Prerequisiti

* Python 3.12 (o compatibile)

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

### Esempi di richiesta (cURL)

#### 1. Spettro NIR
Richiesta di inferenza su uno spettro NIR (parametri spettrali: `wavelength` e `absorbance`, 125 elementi):

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "measurement_type": "nir",
    "tipo di campione": "pesce",
    "specie campione": "polpo",
    "wavelength": [900.0, 906.45, 912.9, "..."],
    "absorbance": [0.3745, 0.9507, 0.7319, "..."]
  }'
```

#### 2. Spettro Raman
Richiesta di inferenza su uno spettro Raman (parametri spettrali: `shift_raman` e `arbitrary_units`):

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "measurement_type": "raman",
    "tipo di campione": "pesce",
    "specie campione": "polpo",
    "shift_raman": [200.0, 210.5, 221.0, "..."],
    "arbitrary_units": [120.5, 340.2, 510.8, "..."]
  }'
```

### Esempio di risposta (JSON)

In caso di successo (HTTP 200 OK), il server restituisce la label del campione con la probabilità che il campione appartenga alla classe predetta (es. `fresco` o `decongelato` per il pesce; `non_csm` o `csm` per la carne):

```json
{
  "label": "decongelato",
  "probability": 0.842
}
```

---

## Codici di stato HTTP

| Codice HTTP | Definizione | Descrizione / Causa | Esempio Body Risposta |
| :--- | :--- | :--- | :--- |
| **`200 OK`** | Successo | Inferenza eseguita con successo | `{"label": "decongelato", "probability": 0.842}` |
| **`400 Bad Request`** | Richiesta non valida | Parametri o combinazioni non gestibili durante l'inferenza | `{"detail": "Combinazione non supportata: ..."}` |
| **`404 Not Found`** | Modello non trovato | Specie supportata ma file del modello non presente su disco | `{"detail": "Modello non trovato"}` |
| **`405 Method Not Allowed`** | Metodo non consentito | Chiamata con metodo HTTP non consentito (es. `GET /predict`) | `{"detail": "Method Not Allowed"}` |
| **`422 Unprocessable Entity`** | Errore di validazione | Campi mancanti, tipi errati, lunghezze errate o valori non conformi | `{"detail": "Payload non valido: ..."}` |
| **`500 Internal Server Error`** | Errore interno | Errore imprevisto durante l'elaborazione o l'inferenza | `{"detail": "Errore interno del server..."}` |
