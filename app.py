from src.customerchurn.exception import CustomException
from src.customerchurn.logger import logging
from src.customerchurn.components.data_ingestion import DataIngestion, DataIngestionConfig
import sys


if __name__ == "__main__":

    try:
        data_ingestion = DataIngestion(config_path="configs/config.yaml")
        train_path, test_path = data_ingestion.initiate_data_ingestion()
        logging.info("Data Ingestion Complete")

    except Exception as e:
        raise CustomException(e,sys)