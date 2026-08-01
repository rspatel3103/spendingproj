"""FastAPI endpoint test for GET /forecast using httpx.AsyncClient.

Exercises the real app (via ASGI transport, in-process -- no network,
no server subprocess) against seeded data in the isolated test DB.
GET /forecast never calls the LLM (forecast_cashflow() is pure
arithmetic), so nothing needs to be mocked here.
"""

from datetime import date
from decimal import Decimal

import httpx

from app.db.models import Transaction
from app.main import app


async def test_get_forecast_returns_seeded_balance_curve(test_sessionmaker):
    today = date.today()

    async with test_sessionmaker() as session:
        session.add_all(
            [
                Transaction(
                    account_id="acct_test",
                    posted_date=today,
                    amount=Decimal("3000.00"),
                    raw_description="PAYROLL",
                    is_recurring=True,
                ),
                Transaction(
                    account_id="acct_test",
                    posted_date=today,
                    amount=Decimal("-50.00"),
                    raw_description="GROCERIES",
                    is_recurring=False,
                ),
            ]
        )
        await session.commit()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/forecast", params={"horizon": 30})

    assert response.status_code == 200
    body = response.json()

    assert body["horizon"] == 30
    assert Decimal(body["starting_balance"]) == Decimal("2950.00")
    assert len(body["balance_curve"]) == 30
    assert body["explanation"]
    assert "lowest_point" in body and "lowest_point_date" in body


async def test_get_forecast_rejects_invalid_horizon(test_sessionmaker):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/forecast", params={"horizon": 0})

    assert response.status_code == 422
