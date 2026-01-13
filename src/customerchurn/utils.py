import yaml
import os
from src.customerchurn.exception import CustomException
import sys
from pathlib import Path
import pickle

def read_yaml(path:str) -> dict:
    try:
        p=Path(path)

        if not p.exists():
            raise FileNotFoundError(f"YAML file not found: {p.resolve()}")

        with p.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if data is None:
            raise ValueError(f"YAML file is empty or invalid: {p.resolve()}")

        return data
    except Exception as e:
        raise CustomException(e,sys)
    


def save_object(file_path:str,obj) -> None:
    os.makedirs(os.path.dirname(file_path),exist_ok=True)
    with open(file_path,"wb") as f:
        pickle.dump(obj,f)
    

def load_object(file_path:str):
    with open(file_path,"rb") as f:
        return pickle.load(f)