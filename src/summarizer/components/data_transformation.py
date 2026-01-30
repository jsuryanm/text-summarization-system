from src.summarizer.config.configuration import ConfigurationManager
from src.summarizer.entity.config_entity import DataTransformationConfig
from src.summarizer.logging.logger import logger
from src.summarizer.utils.common import create_directories

from datasets import load_from_disk
from transformers import AutoTokenizer 
import os

class DataTransformation:
    def __init__(self,
                 config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)
        
        create_directories([self.config.root_dir])
    
    def _tokenize_batch(self,batch):
        inputs = self.tokenizer(batch['article'],
                                max_length=1024,
                                padding='max_length',
                                truncation=True)
        
        labels = self.tokenizer(batch['highlights'],
                                max_length=128,
                                truncation=True,
                                padding='max_length')
        
        inputs['labels'] = labels['input_ids']
        return inputs 
    
    def convert(self):
        logger.info(f"Loading the dataset from disk")
        dataset = load_from_disk(self.config.data_path)

        logger.info("Tokenizing the dataset")
        
        tokenized_dataset = dataset.map(self._tokenize_batch,
                                        batched=True,
                                        remove_columns=dataset['train'].column_names,
                                        num_proc=1)

        output_path = os.path.join(self.config.root_dir,"cnn_dailymail")
        logger.info(f"Saving the tokenized dataset to {output_path}")
        tokenized_dataset.save_to_disk(output_path)