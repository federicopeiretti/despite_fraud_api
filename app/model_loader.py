import os
import threading
import tensorflow as tf

# Compatibility workaround for Keras models with legacy initializer configurations
try:
    from keras.src.initializers.initializer import Initializer

    @classmethod
    def _safe_initializer_from_config(cls, config):
        if isinstance(config, dict):
            config = {k: v for k, v in config.items() if k not in ("input_axes", "output_axes")}
        return cls(**config)

    Initializer.from_config = _safe_initializer_from_config
except Exception:
    pass

class ModelManager:
    def __init__(self):
        self._models = {}
        self._lock = threading.Lock()

    def get_model(self, instrument: str, animal: str, species: str, path: str):
        key = (instrument, animal, species)
        
        # Double-checked locking to avoid lock overhead once model is loaded
        if key in self._models:
            return self._models[key]
            
        with self._lock:
            if key in self._models:
                return self._models[key]
                
            if not os.path.exists(path):
                raise FileNotFoundError(f"Modello non trovato per {key}: {path}")
                
            print(f"Caricamento lazy del modello per {key} da {path}...")
            # Load the Keras ensemble model
            model = tf.keras.models.load_model(path)
            self._models[key] = model
            return model

model_manager = ModelManager()
