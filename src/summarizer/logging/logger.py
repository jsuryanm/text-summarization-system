import sys 
import logging 
import os 
from datetime import datetime 

logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

log_dir = "logs"
os.makedirs(log_dir,exist_ok=True)

timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M")
logs_filepath = os.path.join(log_dir,f"{timestamp}_running_logs.log")

logging.basicConfig(level=logging.INFO,
                    format=logging_str,
                    handlers=[
                        logging.FileHandler(logs_filepath),
                        logging.StreamHandler(sys.stdout)
                    ])

logger = logging.getLogger("text_summarizer_logger")