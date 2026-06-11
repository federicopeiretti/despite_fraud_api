import math
from typing import List
from pydantic import BaseModel, Field, field_validator, model_validator
from app.config import SUPPORTED_INSTRUMENTS, SUPPORTED_ANIMALS, SUPPORTED_SPECIES, SPECTRUM_LENGTHS

class PredictRequest(BaseModel):
    measurement_type: str
    tipo_di_campione: str = Field(..., alias="tipo di campione")
    specie_campione: str = Field(..., alias="specie campione")
    wavelength: List[float]
    absorbance: List[float]

    model_config = {
        "populate_by_name": True,
        "extra": "ignore"
    }

    @field_validator("measurement_type", "tipo_di_campione", "specie_campione", mode="before")
    @classmethod
    def clean_text_field(cls, v):
        if isinstance(v, str):
            return v.strip().lower()
        return v

    @field_validator("wavelength", "absorbance")
    @classmethod
    def validate_finite_and_nonempty(cls, v):
        if not v:
            raise ValueError("Il vettore non puo' essere vuoto.")
        for x in v:
            if not isinstance(x, (int, float)) or isinstance(x, bool):
                raise ValueError("Tutti i valori devono essere numerici.")
            if not math.isfinite(x):
                raise ValueError("Tutti i valori devono essere finiti (no NaN o Inf).")
        return v

    @model_validator(mode="after")
    def validate_cross_fields(self) -> "PredictRequest":
        m_type = self.measurement_type
        t_campione = self.tipo_di_campione
        s_campione = self.specie_campione
        wl = self.wavelength
        abs_vec = self.absorbance

        # Check support
        if m_type not in SUPPORTED_INSTRUMENTS:
            raise ValueError(f"Strumento '{m_type}' non supportato. Supportati: {list(SUPPORTED_INSTRUMENTS)}")
        if t_campione not in SUPPORTED_ANIMALS:
            raise ValueError(f"Tipo di campione '{t_campione}' non supportato. Supportati: {list(SUPPORTED_ANIMALS)}")
        if s_campione not in SUPPORTED_SPECIES:
            raise ValueError(f"Specie '{s_campione}' non supportata. Supportati: {list(SUPPORTED_SPECIES)}")

        # Check lengths match
        if len(wl) != len(abs_vec):
            raise ValueError("I vettori wavelength e absorbance devono avere la stessa lunghezza.")

        # Check expected length for instrument
        expected_len = SPECTRUM_LENGTHS.get(m_type)
        if expected_len is not None and len(abs_vec) != expected_len:
            raise ValueError(f"La lunghezza dello spettro ({len(abs_vec)}) non e' compatibile con lo strumento '{m_type}' (atteso: {expected_len}).")

        return self

class PredictResponse(BaseModel):
    label: str
    probability: float
