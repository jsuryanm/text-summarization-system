from datasets import load_dataset 
from src.summarizer.logging.logger import logger 
from src.summarizer.entity.config_entity import DataIngestionConfig

class DataIngestion: 
    def __init__(self,config: DataIngestionConfig):
        self.config = config
    
    def ingest(self):
        if self.config.dataset.source != "huggingface":
            raise ValueError("Only HuggingFace datasets are supported")

        logger.info(f"Loaing dataset {self.config.dataset.hf_repo}"
                    f"(version={self.config.dataset.version})")
        
        dataset = load_dataset(self.config.dataset.hf_repo,
                               self.config.dataset.version)
        
        dataset.save_to_disk(self.config.local_data_dir)
        logger.info(f"Dataset saved to {self.config.local_data_dir}")
