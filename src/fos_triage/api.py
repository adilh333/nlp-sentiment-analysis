"""HTTP triage endpoint.

    uvicorn fos_triage.api:app --reload
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from fos_triage import __version__
from fos_triage.service import TriageService

app = FastAPI(
    title="UK financial complaint triage",
    version=__version__,
    summary="Product coding, uphold likelihood and a short case note from a complaint narrative.",
)


class ComplaintIn(BaseModel):
    text: str = Field(min_length=40, max_length=20000)
    business_name: str | None = Field(default=None, max_length=200)


def get_service() -> TriageService:
    service = getattr(app.state, "service", None)
    if service is None:
        try:
            service = TriageService.load()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail="Models are not trained yet") from exc
        app.state.service = service
    return service


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.post("/triage")
def triage(payload: ComplaintIn, service: TriageService = Depends(get_service)) -> dict:
    return service.triage(payload.text, payload.business_name)
