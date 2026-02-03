import os 
from dotenv import load_dotenv
from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
from huggingface_hub import login 

from src.summarizer.logging.logger import logger 
from src.summarizer.entity.config_entity import ModelPusherConfig

load_dotenv()

class ModelPusher:
    def __init__(self,config: ModelPusherConfig):
        self.config = config

    def push(self):
        logger.info("Starting model push to HuggingFace Hub")

        hf_token = os.getenv("HF_TOKEN")
        if hf_token is None: 
            raise RuntimeWarning("Set the hf_token to login")
        
        login(token=hf_token)
        logger.info("Loading model and tokenizer from artifacts")
        
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_dir)
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_dir) 
        
        logger.info(f"Pushing to HuggingFace repo:{self.config.hf_repo_id}")
        
        model.push_to_hub(self.config.hf_repo_id, private=self.config.hf_private)
        tokenizer.push_to_hub(self.config.hf_repo_id)

        logger.info("Model push completed successfully")