import os 
from box.exceptions import BoxValueError
import yaml
from src.textSummarizer.logging import logger
from ensure import ensure_annotations
from box import ConfigBox 
from typing import Any
from pathlib import Path

# @ensure_annotations
# def read_yaml(path_to_yaml: str) -> ConfigBox:
#     """
#     Reads YAML file and returns 
    
#     Args: 
#         path_to_yaml (str): path like input  
#     Raises: 
#         ValueError: if yaml file is empty 
#         e: empty file 
        
#     Returns:
#         configBox: ConfigBox type

#     """
#     try:
#         with open(path_to_yaml, 'r') as yaml_file:
#             content = yaml.safe_load(yaml_file)
#             logger.info(f"YAML file: {path_to_yaml} loaded successfully")
#             return ConfigBox(content)
#     except BoxValueError as e:
#         raise ValueError(f"YAML file: {path_to_yaml} is empty") 
#     except Exception as e :
#         raise Exception(f"Error occurred while reading YAML file: {path_to_yaml}") from e


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    
def create_directories(path_to_directories: list, verbose: bool = True):
    """
    Creates list of directories
    
    Agrs: 
        path_to_directories (list): list of path of directories 
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Deaults to False. 
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose: 
            logger.info(f"Directory created: {path}")

@ ensure_annotations 
def get_size(path:Path) -> str:
    """ get size in KB
    
    Args: 
        path (Path): path of the file 
    
    Returns: 
        str: size of the file in KB 
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"{size_in_kb} KB"