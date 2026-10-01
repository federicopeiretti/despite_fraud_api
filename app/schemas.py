import math
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator, AliasChoices
from app.config import SUPPORTED_INSTRUMENTS, SUPPORTED_ANIMALS, SUPPORTED_SPECIES, SUPPORTED_SPECIES_BY_ANIMAL, SPECTRUM_LENGTHS

class PredictRequest(BaseModel):
    measurement_type: str
    tipo_di_campione: str = Field(..., alias="tipo di campione")
    specie_campione: str = Field(..., alias="specie campione")
    
    # Parametri specifici per strumento NIR
    wavelength: Optional[List[float]] = None
    absorbance: Optional[List[float]] = None
    
    # Parametri specifici per strumento Raman
    shift_raman: Optional[List[float]] = Field(None, validation_alias=AliasChoices("shift_raman", "shift raman", "shift-raman"))
    arbitrary_units: Optional[List[float]] = Field(None, validation_alias=AliasChoices("arbitrary_units", "arbitrary units", "arbitrary-units"))

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

    @field_validator("wavelength", "absorbance", "shift_raman", "arbitrary_units")
    @classmethod
    def validate_finite_and_nonempty(cls, v):
        if v is None:
            return v
        if not v:
            raise ValueError("Il vettore non può essere vuoto.")
        for x in v:
            if not isinstance(x, (int, float)) or isinstance(x, bool):
                raise ValueError("Tutti i valori devono essere numerici.")
            if not math.isfinite(x):
                raise ValueError("Tutti i valori devono essere finiti. Non sono ammessi valori NaN o inf.")
        return v

    @model_validator(mode="after")
    def validate_cross_fields(self) -> "PredictRequest":
        m_type = self.measurement_type
        t_campione = self.tipo_di_campione
        s_campione = self.specie_campione

        # Check support
        if m_type not in SUPPORTED_INSTRUMENTS:
            raise ValueError(f"Strumento '{m_type}' non supportato. Scegliere tra: {list(SUPPORTED_INSTRUMENTS)}")
        if t_campione not in SUPPORTED_ANIMALS:
            raise ValueError(f"Tipo di campione '{t_campione}' non supportato. Scegliere tra: {list(SUPPORTED_ANIMALS)}")
        
        allowed_species = SUPPORTED_SPECIES_BY_ANIMAL.get(t_campione, SUPPORTED_SPECIES)
        if s_campione not in allowed_species:
            raise ValueError(f"Specie '{s_campione}' non supportata per il tipo di campione '{t_campione}'. Scegliere tra: {list(allowed_species)}")

        # Check instrument-specific payload requirements
        if m_type == "nir":
            if self.wavelength is None or self.absorbance is None:
                raise ValueError("Per lo strumento 'nir' sono richiesti i parametri 'wavelength' e 'absorbance'.")
            if len(self.wavelength) != len(self.absorbance):
                raise ValueError("I vettori wavelength e absorbance devono avere la stessa lunghezza.")

            expected_len = SPECTRUM_LENGTHS.get("nir")
            if expected_len is not None and len(self.absorbance) != expected_len:
                raise ValueError(f"La lunghezza dello spettro ({len(self.absorbance)}) non è compatibile con lo strumento '{m_type}'. Lunghezza attesa: {expected_len}.")

        elif m_type == "raman":
            if self.shift_raman is None or self.arbitrary_units is None:
                raise ValueError("Per lo strumento 'raman' sono richiesti i parametri 'shift_raman' e 'arbitrary_units'.")
            if len(self.shift_raman) != len(self.arbitrary_units):
                raise ValueError("I vettori shift_raman e arbitrary_units devono avere la stessa lunghezza.")

            expected_len = SPECTRUM_LENGTHS.get("raman")
            if expected_len is not None and len(self.arbitrary_units) != expected_len:
                raise ValueError(f"La lunghezza dello spettro ({len(self.arbitrary_units)}) non è compatibile con lo strumento '{m_type}'. Lunghezza attesa: {expected_len}.")

        return self

class PredictResponse(BaseModel):
    label: str
    probability: float

    @field_validator("probability")
    @classmethod
    def round_probability(cls, v: float) -> float:
        return round(float(v), 3)
