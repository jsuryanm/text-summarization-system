
from src.summarizer.constants.constant import *
from src.summarizer.utils.common import read_yaml,create_directories
from src.summarizer.entity.config_entity import (DataIngestionConfig,
                                                 DatasetConfig,
                                                 DataValidationConfig,
                                                 DataTransformationConfig,
                                                 ModelTrainerConfig,
                                                 ModelEvaluationConfig,
                                                 ModelPusherConfig,
                                                 InferenceConfig)

class ConfigurationManager:
    def __init__(self,
                 config_filepath=CONFIG_FILE_PATH,
                 params_filepath=PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        ingestion_cfg = self.config.data_ingestion
        dataset_cfg = self.config.dataset

        create_directories([ingestion_cfg.root_dir])

        dataset_cfg = DatasetConfig(
            name=dataset_cfg.name,
            source=dataset_cfg.source,
            hf_repo=dataset_cfg.hf_repo,
            version=dataset_cfg.version,
            text_column=dataset_cfg.text_column,
            summary_column=dataset_cfg.summary_column
        )

        return DataIngestionConfig(root_dir=Path(ingestion_cfg.root_dir),
                                   local_data_dir=Path(ingestion_cfg.local_data_dir),
                                   dataset=dataset_cfg)

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation

        create_directories([config.root_dir])
        
        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            ALL_REQUIRED_FIELDS=config.ALL_REQUIRED_FIELDS
        )

        return data_validation_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(root_dir=config.root_dir,
                                                              data_path=config.data_path,
                                                              tokenizer_name=config.tokenizer_name)
        
        return data_transformation_config
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.model_trainer

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(root_dir=config.root_dir,
                                                  data_path=config.data_path,
                                                  model_checkpoint=config.model_checkpoint,
                                                  num_train_epochs=params.num_train_epochs,
                                                  learning_rate=params.learning_rate,
                                                  warmup_steps=params.warmup_steps,
                                                  per_device_train_batch_size=params.per_device_train_batch_size,
                                                  per_device_eval_batch_size=params.per_device_eval_batch_size,
                                                  gradient_accumulation_steps=params.gradient_accumulation_steps,
                                                  weight_decay=params.weight_decay,
                                                  optim=params.optim,
                                                  lr_scheduler_type=params.lr_scheduler_type,
                                                  use_cpu=params.use_cpu,
                                                  use_bf16=params.bf16,
                                                  seed=params.seed,
                                                  data_seed=params.data_seed,
                                                  tf32=params.tf32,
                                                  gradient_checkpointing=params.gradient_checkpointing,
                                                  dataloader_pin_memory=params.dataloader_pin_memory,
                                                  dataloader_num_workers=params.dataloader_num_workers,
                                                  torch_empty_cache_steps=params.torch_empty_cache_steps,
                                                  logging_strategy=params.logging_strategy,
                                                  logging_steps=params.logging_steps,
                                                  eval_strategy=params.eval_strategy,
                                                  eval_steps=params.eval_steps,
                                                  save_strategy=params.save_strategy,
                                                  save_steps=params.save_steps,
                                                  report_to=params.report_to,
                                                  load_best_model_at_end=params.load_best_model_at_end,
                                                  generation_num_beams=params.generation_num_beams,
                                                  generation_max_length=params.generation_max_length,
                                                  predict_with_generate=params.predict_with_generate,
                                                  train_fraction=params.train_fraction,
                                                  eval_fraction=params.eval_fraction,
                                                  max_grad_norm=params.max_grad_norm,
                                                  torch_compile=params.torch_compile)
        
        return model_trainer_config
    
    def get_model_eval_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation

        create_directories([config.root_dir])

        model_eval_config = ModelEvaluationConfig(root_dir=config.root_dir,
                                                  data_path=config.data_path,
                                                  model_path=config.model_path,
                                                  tokenizer_path=config.tokenizer_path,
                                                  metric_file_name=config.metric_file_name)
        return model_eval_config
    
    def get_model_pusher_config(self) -> ModelPusherConfig:
        config = self.config.model_pusher
        model_pusher_config = ModelPusherConfig(model_dir=config.model_dir,
                                                tokenizer_dir=config.tokenizer_dir,
                                                hf_repo_id=config.hf_repo_id,
                                                hf_private=config.hf_private)
        return model_pusher_config
    
    def get_inference_config(self) -> InferenceConfig:
        config = self.config.inference

        if not config.hf_repo_id:
            raise ValueError("HuggingFace repository must set for inference")
        
        infer_config = InferenceConfig(hf_repo_id=config.hf_repo_id)
        return infer_config