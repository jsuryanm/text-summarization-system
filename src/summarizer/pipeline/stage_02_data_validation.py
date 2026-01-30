from src.summarizer.config.configuration import ConfigurationManager
from src.summarizer.components.data_validation import DataValidation

class DataValidationTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        try:
            config = ConfigurationManager()
            data_validation_config = config.get_data_validation_config()
            data_validation = DataValidation(data_validation_config)
            data_validation.validate_all_files_exist()
            
        except Exception as e:
            raise e