from __future__ import annotations

from pathlib import Path
from pydantic import BaseModel
from pm_workflows.working_target import WorkingEnvironment
from pm.utils import yaml_util


def load_yaml[T: BaseModel](path: str | Path, schema: type[T]) -> T:
    return schema.model_validate(yaml_util.safe_load(Path(path).read_text(encoding="utf-8")))


def read_request[T: BaseModel](environment: WorkingEnvironment, schema: type[T]) -> T:
    return schema.model_validate(yaml_util.safe_load(environment.read_text("inbox/request.yaml")))


def write_report(environment: WorkingEnvironment, report: BaseModel) -> None:
    environment.write_text("scratch/report.yaml", yaml_util.safe_dump(
        report.model_dump(mode="json"), sort_keys=False, allow_unicode=True))
    environment.move("scratch/report.yaml", "outbox/report.yaml")
