from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml


@dataclass
class ModelConfig:
    name: str
    providers: List[str]


@dataclass
class ClusterConfig:
    eps: float
    min_samples: int


@dataclass
class ConfidenceConfig:
    threshold: float


@dataclass
class VisualizationConfig:
    enabled: bool


@dataclass
class LoggingConfig:
    level: str


@dataclass
class Config:
    dataset_path: Path
    output_path: Path
    report_path: Path

    model: ModelConfig
    clustering: ClusterConfig
    confidence: ConfidenceConfig
    visualization: VisualizationConfig
    logging: LoggingConfig


def load_config(path: str = "config.yaml") -> Config:
    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    return Config(
        dataset_path=Path(data["dataset_path"]),
        output_path=Path(data["output_path"]),
        report_path=Path(data["report_path"]),
        model=ModelConfig(**data["model"]),
        clustering=ClusterConfig(**data["clustering"]),
        confidence=ConfidenceConfig(**data["confidence"]),
        visualization=VisualizationConfig(**data["visualization"]),
        logging=LoggingConfig(**data["logging"]),
    )