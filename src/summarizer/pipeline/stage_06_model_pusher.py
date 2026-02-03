from src.summarizer.config.configuration import ConfigurationManager
from src.summarizer.components.model_pusher import ModelPusher

class ModelPusherPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        pusher_config = config.get_model_pusher_config()

        model_pusher = ModelPusher(pusher_config)
        model_pusher.push()

