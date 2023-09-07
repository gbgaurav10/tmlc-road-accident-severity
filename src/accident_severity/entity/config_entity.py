import os, sys
from pathlib import Path 
from dataclasses import dataclass
 

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path 
    source_url: str 
    local_data_file: Path 
    unzip_dir: Path


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path 
    data_path: Path


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path 
    train_data_path: Path 
    test_data_path: Path 
    model_name: str 
    n_estimators: int
    max_depth: int 
    learning_rate: float
    target_column: str


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path 
    test_data_path: Path 
    model_path: Path 
    all_params: dict 
    metric_file_name: Path 
    target_column: str
    model_name: Path
