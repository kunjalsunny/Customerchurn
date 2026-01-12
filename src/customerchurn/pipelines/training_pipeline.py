from src.customerchurn.exception import CustomException
from src.customerchurn.logger import logging
import sys

from src.customerchurn.components.data_ingestion import DataIngestion
from src.customerchurn.components.data_validation import DataValidation


class TrainPipeline():
    def run(self):
        try:
            logging.info("=====Train Pipeline Started=======")

            ingestion = DataIngestion(config_path="configs/config.yaml")
            train_path, test_path = ingestion.initiate_data_ingestion()


            logging.info(f"Ingestion Completed: {train_path} | {test_path}")
            
            # Validation
            validator = DataValidation()
            report_path = validator.initiate_data_validation(train_path,test_path)

            logging.info(f"Validation Passed. Report: {report_path}")
            logging.info("=======Train Pipeline Completed=======")

            return report_path            


        except Exception as e:
            raise CustomException(e,sys)


if __name__ == "__main__":
    TrainPipeline().run()