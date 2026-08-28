import numpy as np
from scipy.signal import savgol_filter

def snv(X):
    """
    Standard Normal Variate (SNV) normalization.
    """
    mean = np.mean(X, axis=1, keepdims=True)
    std = np.std(X, axis=1, keepdims=True)
    # Avoid division by zero
    std = np.where(std == 0, 1e-8, std)
    return (X - mean) / std

def preprocess_nir(absorbance: list[float]) -> np.ndarray:
    """
    Apply Savitzky-Golay filter (window_length=9, polyorder=2, deriv=2)
    followed by SNV normalization.
    Expects a single spectrum as a list of floats.
    Returns a 2D NumPy array of shape (1, features).
    """
    X = np.array([absorbance], dtype=float)  # Shape: (1, 125)
    X_savgol = savgol_filter(X, window_length=9, polyorder=2, deriv=2, axis=1)
    return snv(X_savgol)

# Dispatcher mapping instrument to its preprocessing function
PREPROCESSORS = {
    "nir": preprocess_nir
    # Raman or SERS preprocessing can be added here in the future
}

def preprocess_spectrum(instrument: str, absorbance: list[float]) -> np.ndarray:
    """
    Dispatches preprocessing based on the instrument type.
    """
    preprocessor = PREPROCESSORS.get(instrument.lower())
    if not preprocessor:
        raise ValueError(f"Nessuna funzione di preprocessing configurata per lo strumento: {instrument}")
    return preprocessor(absorbance)
