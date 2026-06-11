import os

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Centralized configuration of supported parameters
SUPPORTED_INSTRUMENTS = {"nir"}
SUPPORTED_ANIMALS = {"pesce"}
SUPPORTED_SPECIES = {"polpo", "moscardino", "seppia"}

# Expected dimensions for each instrument (for validation)
SPECTRUM_LENGTHS = {
    "nir": 125
}

# Mapping class indices to output labels
CLASSES_MAPPING = {
    0: "fresco",
    1: "decongelato"
}

# Centralized paths to the ensemble models
# Points to the sibling directory 'octopus_nir_classification'
MODEL_PATHS = {
    ("nir", "pesce", "polpo"): os.path.abspath(
        os.path.join(BASE_DIR, "..", "octopus_nir_classification", "model", "cnn", "cnn_ensemble.keras")
    )
    # Future models for moscardino, seppia or other instruments/animals can be added here
}
