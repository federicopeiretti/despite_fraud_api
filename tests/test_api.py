import numpy as np
from fastapi.testclient import TestClient
from app.main import app
from app.model_loader import model_manager

client = TestClient(app)

def get_valid_payload():
    # A valid payload with a random spectrum of length 125
    np.random.seed(42)
    absorbance = np.random.rand(125).tolist()
    wavelength = np.linspace(900, 1700, 125).tolist()
    return {
        "measurement_type": "NIR",
        "tipo di campione": "pesce",
        "specie campione": "polpo",
        "wavelength": wavelength,
        "absorbance": absorbance
    }

def get_valid_raman_payload():
    np.random.seed(42)
    arbitrary_units = np.random.rand(100).tolist()
    shift_raman = np.linspace(200, 3000, 100).tolist()
    return {
        "measurement_type": "raman",
        "tipo di campione": "pesce",
        "specie campione": "polpo",
        "shift_raman": shift_raman,
        "arbitrary_units": arbitrary_units
    }

def test_validation_errors():
    # 1. Test unsupported measurement type
    payload = get_valid_payload()
    payload["measurement_type"] = "FTIR"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "strumento" in response.json()["detail"].lower()

    # 2. Test invalid animal type
    payload = get_valid_payload()
    payload["tipo di campione"] = "verdura"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "tipo di campione" in response.json()["detail"].lower()

    # 3. Test unsupported species
    payload = get_valid_payload()
    payload["specie campione"] = "tonno"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "specie" in response.json()["detail"].lower()

    # Test species not valid for the specific animal type (e.g. wurstel for pesce)
    payload_mismatch = get_valid_payload()
    payload_mismatch["tipo di campione"] = "pesce"
    payload_mismatch["specie campione"] = "wurstel"
    response_mismatch = client.post("/predict", json=payload_mismatch)
    assert response_mismatch.status_code == 422
    assert "specie" in response_mismatch.json()["detail"].lower()

    # 4. Test NIR missing required fields (wavelength/absorbance)
    payload = get_valid_payload()
    del payload["wavelength"]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "wavelength" in response.json()["detail"].lower()

    # 5. Test NIR wavelength and absorbance length mismatch
    payload = get_valid_payload()
    payload["wavelength"] = payload["wavelength"][:-1]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "lunghezza" in response.json()["detail"].lower() or "vettori" in response.json()["detail"].lower()

    # 6. Test NIR spectrum length not matching 125
    payload = get_valid_payload()
    payload["wavelength"] = list(range(100))
    payload["absorbance"] = list(range(100))
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "lunghezza dello spettro" in response.json()["detail"].lower()

    # 7. Test Raman missing required fields
    payload_raman = get_valid_raman_payload()
    del payload_raman["arbitrary_units"]
    response = client.post("/predict", json=payload_raman)
    assert response.status_code == 422
    assert "arbitrary_units" in response.json()["detail"].lower()

    # 8. Test Raman length mismatch
    payload_raman = get_valid_raman_payload()
    payload_raman["shift_raman"] = payload_raman["shift_raman"][:-2]
    response = client.post("/predict", json=payload_raman)
    assert response.status_code == 422
    assert "lunghezza" in response.json()["detail"].lower() or "vettori" in response.json()["detail"].lower()

    # 9. Test infinite values
    payload = get_valid_payload()
    payload["absorbance"][0] = "inf"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "finiti" in response.json()["detail"].lower()

def test_supported_species_missing_model():
    # NIR with Seppia: supported in config, but model file does not exist
    payload = get_valid_payload()
    payload["specie campione"] = "seppia"
    response = client.post("/predict", json=payload)
    assert response.status_code == 404
    assert "modello non trovato" in response.json()["detail"].lower()

    # Raman with Polpo: supported schema, but model file does not exist yet
    payload_raman = get_valid_raman_payload()
    response_raman = client.post("/predict", json=payload_raman)
    assert response_raman.status_code == 404
    assert "modello non trovato" in response_raman.json()["detail"].lower()

    # NIR with Wurstel (carne): supported in config, but model file does not exist yet
    payload_carne = get_valid_payload()
    payload_carne["tipo di campione"] = "carne"
    payload_carne["specie campione"] = "wurstel"
    response_carne = client.post("/predict", json=payload_carne)
    assert response_carne.status_code == 404
    assert "modello non trovato" in response_carne.json()["detail"].lower()

def test_polpo_lazy_loading_and_prediction():
    # Make sure cache is clean for the key
    key = ("nir", "pesce", "polpo")
    if key in model_manager._models:
        del model_manager._models[key]

    payload = get_valid_payload()

    # 1. Send the first request (which should lazy-load the model)
    assert key not in model_manager._models
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert key in model_manager._models  # Loaded into memory

    # 2. Check structure
    data = response.json()
    assert "label" in data
    assert "probability" in data
    assert len(data) == 2  # Contain ONLY label and probability

    # 3. Check types and ranges
    assert data["label"] in ["fresco", "decongelato"]
    assert isinstance(data["probability"], float)
    assert 0.5 <= data["probability"] <= 1.0
    # Verifica massimo 3 cifre decimali
    decimal_part = str(data["probability"]).split(".")[1] if "." in str(data["probability"]) else ""
    assert len(decimal_part) <= 3

    # 4. Send second request (should use cached model, no reload)
    response_cached = client.post("/predict", json=payload)
    assert response_cached.status_code == 200
    assert response_cached.json() == data
