import os
import sys
from src.custom_exception import CustomException
from src.custom_logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    """Data Ingestion Configuration"""
    train_data_path:str=os.path.join('resources','train.csv')
    test_data_path:str=os.path.join('resources','test.csv')
    raw_data_path:str=os.path.join('resources','data.csv')

class DataIngestion:
    """Data Ingestion Class"""
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        """Method to initiate data ingestion"""
        logging.info("Data Ingestion method starts")
        try:
            df=pd.read_csv('data/stud.csv')
            logging.info("Dataset read as pandas dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("Train test split initiated")

            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of data is completed")
            return self.ingestion_config

        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    obj=DataIngestion()
    obj.initiate_data_ingestion()