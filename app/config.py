import os

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Authentication
API_BEARER_TOKEN = os.getenv("API_BEARER_TOKEN", "despite_fraud_secret_token_2026")

# Centralized configuration of supported parameters
SUPPORTED_INSTRUMENTS = {"nir", "raman"}
SUPPORTED_ANIMALS = {"pesce", "carne"}
SUPPORTED_SPECIES = {"polpo", "moscardino", "seppia", "wurstel"}

SUPPORTED_SPECIES_BY_ANIMAL = {
    "pesce": {"polpo", "moscardino", "seppia"},
    "carne": {"wurstel"}
}

# Expected dimensions for each instrument (for validation)
SPECTRUM_LENGTHS = {
    "nir": 125
}

# Mapping class indices to output labels by animal type
CLASSES_MAPPING = {
    "pesce": {
        0: "fresco",
        1: "decongelato"
    },
    "carne": {
        0: "non_csm",
        1: "csm"
    }
}

# Centralized paths to the ensemble models
MODEL_PATHS = {
    ("nir", "pesce", "polpo"): os.path.abspath(
        os.path.join(BASE_DIR, "models", "octopus", "cnn_ensemble.keras")
    )
    # Future models for moscardino, seppia, wurstel or other instruments/animals can be added here
}
