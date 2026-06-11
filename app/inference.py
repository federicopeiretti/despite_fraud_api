import numpy as np
from app.config import MODEL_PATHS, CLASSES_MAPPING
from app.model_loader import model_manager
from app.preprocessing import preprocess_spectrum

def run_inference(
    measurement_type: str,
    tipo_di_campione: str,
    specie_campione: str,
    absorbance: list[float]
) -> tuple[str, float]:
    """
    Coordinates the full inference pipeline:
    1. Look up configured model path.
    2. Retrieve model via lazy loader (singleton ModelManager).
    3. Preprocess absorbance values.
    4. Reshape data dynamically based on model requirements (2D vs 3D for CNN).
    5. Run prediction and map to labels.
    """
    key = (measurement_type.lower(), tipo_di_campione.lower(), specie_campione.lower())
    
    # 1. Resolve model path
    model_path = MODEL_PATHS.get(key)
    if not model_path:
        # Check if the combination is theoretically supported but no model exists
        from app.config import SUPPORTED_INSTRUMENTS, SUPPORTED_ANIMALS, SUPPORTED_SPECIES
        if (key[0] in SUPPORTED_INSTRUMENTS and 
            key[1] in SUPPORTED_ANIMALS and 
            key[2] in SUPPORTED_SPECIES):
            raise FileNotFoundError(f"Modello non ancora implementato/trovato per {key}.")
        raise ValueError(f"Combinazione non supportata: {key}")

    # 2. Get the model (loads lazy on first request)
    model = model_manager.get_model(key[0], key[1], key[2], model_path)

    # 3. Preprocess spectrum
    X_preprocessed = preprocess_spectrum(measurement_type, absorbance)  # shape: (1, 125)

    # 4. Shape matching (e.g. 1D CNN expects 3D input: (batch, features, channels))
    if len(model.input_shape) == 3:
        X_input = X_preprocessed.reshape(X_preprocessed.shape[0], X_preprocessed.shape[1], 1)
    else:
        X_input = X_preprocessed

    # 5. Predict
    # Run predict with verbose=0 to avoid stdout pollution
    preds = model.predict(X_input, verbose=0).flatten()
    prob_positive_class = float(preds[0])

    # Map output probability (binary classification sigmoid)
    # prob_positive_class is the probability of class 1 ("decongelato")
    if prob_positive_class >= 0.5:
        predicted_class_idx = 1
        probability = prob_positive_class
    else:
        predicted_class_idx = 0
        probability = 1.0 - prob_positive_class

    label = CLASSES_MAPPING.get(predicted_class_idx, "sconosciuto")
    return label, probability
