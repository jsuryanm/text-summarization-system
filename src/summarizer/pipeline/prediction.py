from src.summarizer.config.configuration import ConfigurationManager
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM 
import torch 

class PredictionPipeline: 
    def __init__(self):
        self.config_mgr = ConfigurationManager()
        self.config = self.config_mgr.get_inference_config()

        if not self.config.hf_repo_id:
            raise ValueError("HuggingFace repository must be set for inference")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.hf_repo_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.hf_repo_id).to(self.device)

        self.model.eval()
    
    def _check_token_length(self, text: str, max_tokens: int = 1024):
        tokens = self.tokenizer(text,truncation=False,add_special_tokens=True)["input_ids"]

        if len(tokens) > max_tokens:
            raise ValueError(
                f"Input too long: {len(tokens)} tokens (max {max_tokens})"
            )


    
    def predict(self, text: str) -> str:
        self._check_token_length(text)
        
        inputs = self.tokenizer(text,
                                return_tensors="pt",
                                truncation=True,
                                max_length=1024).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs,
                                             max_length=128,
                                             num_beams=8,
                                             length_penalty=0.8,
                                             early_stopping=True)

        return self.tokenizer.decode(output_ids[0],skip_special_tokens=True)