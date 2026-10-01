from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from app.schemas import PredictRequest, PredictResponse
from app.inference import run_inference

app = FastAPI(
    title="despite_fraud_API",
    description="REST API per l'inferenza su dati spettroscopici NIR",
    version="1.0.0"
)

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
    response_description="Predizione con label (fresco/decongelato) e probabilita'"
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
