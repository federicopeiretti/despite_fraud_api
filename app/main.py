import os
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from app.schemas import (
    PredictRequest,
    PredictResponse,
    HealthResponse,
    SupportedConfigResponse,
    SampleTypeConfig,
    ModelItemStatus,
    ModelsListResponse
)
from app.config import (
    SUPPORTED_INSTRUMENTS,
    SUPPORTED_SPECIES_BY_ANIMAL,
    SPECTRUM_LENGTHS,
    CLASSES_MAPPING,
    MODEL_PATHS
)
from app.model_loader import model_manager
from app.inference import run_inference
from app.auth import verify_token

app = FastAPI(
    title="despite_fraud_API",
    description="REST API per l'inferenza su dati spettroscopici NIR e Raman",
    version="1.0.0",
    dependencies=[Depends(verify_token)]
)

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Verifica lo stato operativo del servizio e i modelli caricati in memoria",
    response_description="Stato operativo del servizio e modelli caricati in memoria"
)
async def health():
    loaded_keys = [f"{k[0]}:{k[1]}:{k[2]}" for k in model_manager._models.keys()]
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        models_loaded_count=len(loaded_keys),
        models_loaded=loaded_keys
    )

@app.get(
    "/supported-config",
    response_model=SupportedConfigResponse,
    summary="Ottieni la configurazione degli strumenti, tipi di campione e specie supportate",
    response_description="Lista di strumenti, lunghezze spettro, tipi di campione, specie e relative etichette"
)
@app.get("/supported_config", response_model=SupportedConfigResponse, include_in_schema=False)
async def supported_config():
    sample_types = {}
    for animal, species_set in SUPPORTED_SPECIES_BY_ANIMAL.items():
        labels = list(CLASSES_MAPPING.get(animal, {}).values())
        sample_types[animal] = SampleTypeConfig(
            species=sorted(list(species_set)),
            labels=labels
        )
    return SupportedConfigResponse(
        instruments=sorted(list(SUPPORTED_INSTRUMENTS)),
        spectrum_lengths=SPECTRUM_LENGTHS,
        sample_types=sample_types
    )

@app.get(
    "/models",
    response_model=ModelsListResponse,
    summary="Elenca i modelli configurati e il loro stato",
    response_description="Elenco dei modelli con verifica presenza file e stato in memoria"
)
async def get_models():
    model_items = []
    for key, path in MODEL_PATHS.items():
        model_items.append(
            ModelItemStatus(
                measurement_type=key[0],
                tipo_di_campione=key[1],
                specie_campione=key[2],
                is_file_present=os.path.exists(path),
                is_loaded_in_memory=key in model_manager._models
            )
        )
    return ModelsListResponse(models=model_items)

# Custom handler for Pydantic validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    # Build a clean message list
    cleaned_errors = []
    for err in errors:
        loc = " -> ".join([str(x) for x in err.get("loc", []) if x != "body"])
        msg = err.get("msg", "Valore non valido")
        cleaned_errors.append(f"{loc}: {msg}" if loc else msg)
        
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": "Payload non valido: " + "; ".join(cleaned_errors)}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # Log the real exception internally if needed (in a real app),
    # but do not expose it to the user.
    print(f"Unhandled Exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Errore interno del server durante l'elaborazione."}
    )

@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Esegui inferenza su uno spettro NIR o Raman",
    response_description="Predizione con label e probabilitá"
)
async def predict(request: PredictRequest):
    try:
        spectrum_data = request.absorbance if request.measurement_type == "nir" else request.arbitrary_units
        label, probability = run_inference(
            measurement_type=request.measurement_type,
            tipo_di_campione=request.tipo_di_campione,
            specie_campione=request.specie_campione,
            spectrum_values=spectrum_data
        )
        return PredictResponse(label=label, probability=probability)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        print(f"Error during inference execution: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Errore interno durante l'esecuzione dell'inferenza."
        )
