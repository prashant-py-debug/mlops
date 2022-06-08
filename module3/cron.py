from ast import Sub
from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner

DeploymentSpec(
    name="fhv_cron_schedular",
    flow_location="orchestration.py",
    flow_runner = SubprocessFlowRunner(),
    schedule=CronSchedule(
        cron="0 9 15 * *",  # 9 am every 15th on month
        timezone="America/New_York"),
)
