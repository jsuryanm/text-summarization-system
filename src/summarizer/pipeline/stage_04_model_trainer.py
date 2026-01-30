from src.summarizer.components.model_trainer import ModelTrainer
from src.summarizer.config.configuration import ConfigurationManager
from pathlib import Path
from src.summarizer.logging.logger import logger

class ModelTrainerPipeline:
    def __init__(self):
        pass
    
    def main(self):
        try:
            
            config = ConfigurationManager()
            model_trainer_config = config.get_model_trainer_config()
            
            root_dir = Path(model_trainer_config.root_dir) 
            model_dir = root_dir / "model"
            tokenizer_dir = root_dir / "tokenizer"

            if model_dir.exists() and tokenizer_dir.exists():
                logger.info(f"Training skipped: final model and tokenizer already exist in artifacts dir:{root_dir}")
                return 
            
            model_trainer = ModelTrainer(config=model_trainer_config)
            model_trainer.train()
            
        except Exception as e:
            raise e