import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from src.logger import logger
from src.exception import CustomException
from src.utils import Utils


@dataclass
class DataIngestionConfig:
    ## we are using dataclass so no need of adding __init__ method, it will be automatically created by dataclass
    raw_data_path:str = os.path.join('artifacts', 'data_ingestion',"raw.csv")
    train_data_path:str = os.path.join('artifacts', 'data_ingestion', 'train.csv')
    test_data_path = os.path.join('artifacts', 'data_ingestion', 'test.csv')
    
class dataIngestion:
    
    """
    Handles data ingestion from source.
    - Loads dataset
    - Saves raw copy
    - Splits into train and test sets
    """

    def __init__(self):
        self.ingestion_config= DataIngestionConfig()
        self.config = Utils.read_yaml('config/config.yaml')
    
    def initiate_data_ingestion(self):
        
        """
        Loads data, saves raw copy, splits into train/test.
        Returns:tuple: (train_data_path, test_data_path)
        
        """
        logger.info('>>>> data Ingestion Started ' )
        try:
            # Step 1 — Load dataset
            # Replace this with pd.read_csv() for your own dataset
            
            housing = fetch_california_housing(as_frame = True)
            df= housing.frame
            logger.info(f'Datset updated successfully of shape {df.shape}')
            logger.info(f'columns : {df.columns}')
            
            # Step 2 save raw data 
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok = True)
            
            df.to_csv(self.ingestion_config.raw_data_path, index = False)
            logger.info(f' raw data saved at {self.ingestion_config.raw_data_path}')
            
            ## Step 3 split into train and test set
            test_size = self.config['data_ingestion']['test_size']
            random_state = self.config['data_ingestion']['random_state']
            
            train_set, test_set = train_test_split(df, test_size= test_size, random_state= random_state)
            
            train_set.to_csv(self.ingestion_config.train_data_path, index= False)
            test_set.to_csv(self.ingestion_config.test_data_path, index= False)
            
            logger.info(f'Train shape: {train_set.shape} | Test shape: {test_set.shape}')
            logger.info('>>>> Data Ingestion Completed')
            
            return(
                self.ingestion_config.train_data_path, 
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            raise CustomException(e,sys)
            