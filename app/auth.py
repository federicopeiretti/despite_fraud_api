from fastapi import HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.config import API_BEARER_TOKEN

# HTTPBearer with auto_error=False to customize the 401 response
security = HTTPBearer(auto_error=False)

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    """
    Validates the Bearer token provided in the Authorization header.
    Raises HTTP 401 Unauthorized if the token is missing or invalid.
    """
    if not credentials or credentials.scheme.lower() != "bearer" or credentials.credentials != API_BEARER_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token di autenticazione non valido o mancante.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials
