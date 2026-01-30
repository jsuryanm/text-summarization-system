
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DatasetConfig:
    name: str
    source: str
    hf_repo: str
    version: str | None
    text_column: str
    summary_column: str


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    local_data_dir: Path
    dataset: DatasetConfig


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    ALL_REQUIRED_FIELDS: str

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path 
    data_path: Path
    tokenizer_name: Path

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    data_path: Path
    model_checkpoint: Path

    num_train_epochs: int
    learning_rate: float
    warmup_steps: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    weight_decay: float

    optim: str
    lr_scheduler_type: str

    use_cpu: bool
    use_bf16: bool
    tf32: bool
    seed: int
    data_seed: int
    gradient_checkpointing: bool
    dataloader_pin_memory: bool
    dataloader_num_workers: int
    torch_empty_cache_steps: int

    logging_strategy: str
    logging_steps: int
    eval_strategy: str
    eval_steps: int
    save_strategy: str
    save_steps: int

    report_to: str 
    load_best_model_at_end: bool
    generation_num_beams: int
    generation_max_length: int
    predict_with_generate: bool

    train_fraction: float 
    eval_fraction: float
    max_grad_norm: float
    torch_compile: bool

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path 
    data_path: Path 
    model_path: Path
    tokenizer_path: Path
    metric_file_name: Path