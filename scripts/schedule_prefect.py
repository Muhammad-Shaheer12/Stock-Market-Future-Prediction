from __future__ import annotations
import os
from prefect.deployments import Deployment
from prefect.client.orchestration import get_client
from prefect.server.schemas.schedules import CronSchedule
from src.prefect.flows import main_flow


async def register():
    cron = os.getenv("PREFECT_CRON", "0 2 * * *")
    symbol = os.getenv("DEFAULT_SYMBOL", "AAPL")
    dep = Deployment.build_from_flow(
        flow=main_flow,
        name="mh-predictor-daily",
        parameters={"symbol": symbol},
        schedules=[CronSchedule(cron=cron, timezone="UTC")],
        tags=["training", "daily"],
    )
    await dep.apply()
    async with get_client() as client:
        print("Deployment registered:", await client.read_deployment(dep.id))


if __name__ == "__main__":
    import anyio

    anyio.run(register)
