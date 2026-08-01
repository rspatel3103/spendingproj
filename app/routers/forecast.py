"""Forecast router.

GET /forecast?horizon=30 (or 90, or any other day count) runs the
forecaster agent (`app/agents/forecaster.py`) and returns the projected
balance curve, lowest point, and a plain-language explanation.
Synchronous -- no background job, since forecasting is fast (no LLM
call, just arithmetic over already-stored transactions).
"""

from fastapi import APIRouter, Query

from app.agents.forecaster import forecast_cashflow
from app.schemas import ForecastResponse

router = APIRouter(prefix="/forecast", tags=["forecast"])


@router.get("", response_model=ForecastResponse)
async def get_forecast(horizon: int = Query(30, ge=1, le=365)) -> ForecastResponse:
    result = await forecast_cashflow(horizon)
    return ForecastResponse(**result)
