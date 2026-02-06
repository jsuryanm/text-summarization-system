import os 
from box.exceptions import BoxValueError
import yaml 
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path 
from typing import Any

from src.summarizer.logging.logger import logger 


@ensure_annotations
def read_yaml(yaml_filepath: Path) -> ConfigBox:
    try:
        with open(yaml_filepath) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file:{yaml_filepath} loaded successfully")

        if content is None:
            raise BoxValueError("yaml file is empty")

        import os

        def resolve_env_vars(value):
            """
            Resolve ${ENV_VAR} patterns in YAML values
            """
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_key = value[2:-1]
                env_value = os.getenv(env_key)
                if env_value is None:
                    raise ValueError(f"Environment variable '{env_key}' is not set")
                return env_value
            return value

        def recurse(obj):
            if isinstance(obj, dict):
                return {k: recurse(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [recurse(item) for item in obj]
            else:
                return resolve_env_vars(obj)

        resolved_content = recurse(content)
        return ConfigBox(resolved_content)

    except BoxValueError:
        raise ValueError("yaml file is empty")

    except Exception as e:
        raise e


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):  
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")

    
@ensure_annotations 
def get_size(path: Path) -> str:
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"

@ensure_annotations
def training_artifacts_exist(model_dir: str, tokenizer_dir: str) -> bool:
    return os.path.isdir(model_dir) and os.path.isdir(tokenizer_dir)
