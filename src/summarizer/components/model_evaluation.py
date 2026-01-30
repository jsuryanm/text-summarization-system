import torch
import pandas as pd 
from datasets import load_from_disk
import evaluate
from tqdm import tqdm 

from transformers import AutoTokenizer,AutoModelForSeq2SeqLM

from src.summarizer.entity.config_entity import ModelEvaluationConfig
from src.summarizer.logging.logger import logger 
from src.summarizer.utils.common import training_artifacts_exist

class ModelEvaluation:
    def __init__(self,config: ModelEvaluationConfig):
        self.config = config 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _generate_summaries(self,model,tokenizer,texts,batch_size):
        summaries = []

        for i in tqdm(range(0,len(texts),batch_size)):
            batch = texts[i : i + batch_size]

            inputs = tokenizer(batch,
                               truncation=True,
                               padding=True,
                               max_length=1024,
                               return_tensors='pt').to(self.device)
            
            with torch.no_grad():
                generated = model.generate(**inputs,
                                           max_length = 128,
                                           num_beams=4)
                
                decoded =  tokenizer.batch_decode(generated,
                                                  skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=True)
                summaries.extend(decoded)
        
        return summaries
    
    def evaluate(self):
        logger.info("Loading the tokenizer and model")
        
        tokenizer_path = str(self.config.tokenizer_path)
        model_path = str(self.config.model_path) 

        if not training_artifacts_exist(model_path,tokenizer_path):
            raise RuntimeError(f"Missing model and tokenizer artifacts.\n"
                               f"Expected tokenizer at:{tokenizer_path}\n"
                               f"Expected model at:{model_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

        model.to(self.device)
        model.eval()

        logger.info("Loading the test dataset")
        dataset = load_from_disk(self.config.data_path)
        test_ds = dataset["test"].shuffle(seed=42).select(
            range(int(0.1 * len(dataset["test"]))))


        rouge = evaluate.load("rouge")
        predictions = self._generate_summaries(model,
                                               tokenizer,
                                               test_ds['article'],
                                               batch_size=32)
        
        rouge_result = rouge.compute(predictions=predictions,
                                     references=test_ds['highlights'])
        logger.info(f"ROUGE results: {rouge_result}")

        df = pd.DataFrame(
            {k: [v] for k, v in rouge_result.items()}
        )
        df.to_csv(self.config.metric_file_name, index=False)

        logger.info(f"Metrics saved to {self.config.metric_file_name}")
        