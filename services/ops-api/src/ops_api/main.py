from fastapi import FastAPI

from common.settings import get_settings

SERVICE_NAME = "ops-api"

app = FastAPI(title=SERVICE_NAME)


@app.get("/health")
def health() -> dict[str, str]:
    settings = get_settings()
    return {"status": "ok", "service": SERVICE_NAME, "env": settings.app_env}
