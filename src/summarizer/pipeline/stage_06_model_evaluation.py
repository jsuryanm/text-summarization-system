from src.summarizer.components.model_evaluation import ModelEvaluation
from src.summarizer.config.configuration import ConfigurationManager


class ModelEvaluationPipeline:
    def __init__(self):
        pass
    
    def main(self):
        try:
            config = ConfigurationManager()
            model_eval_config = config.get_model_eval_config()
            model_eval = ModelEvaluation(config=model_eval_config)
            model_eval.evaluate()
            
        except Exception as e:
            raise e