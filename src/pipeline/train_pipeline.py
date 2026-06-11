import sys
import os
from src.logger import logger
from src.exception import CustomException
from src.components.data_ingestion import dataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainPipeline:
    """
    Orchestrates the full training pipeline.
    Connects all 3 components end to end.
    
    """
    
    def __init__(self):
        self.data_ingestion = dataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()
        
    def run(self):
        """
        Runs full training pipeline:
        Data Ingestion → Data Transformation → Model Training

        Returns:
            tuple: (best_score, best_model_name, model_report)
        """
        try:
            logger.info('='*50)
            logger.info('TRAINING PIPELINE STARTED')
            logger.info("="*50)
            
            #Step 1 - Data Ingestion
            logger.info('Step 1 : Data Ingestion')
            train_path, test_path = self.data_ingestion.initiate_data_ingestion()
            
            #Step 2 - DATA TRANSFORMATION
            logger.info('STEP 2 - DATA TRANSFORMATION')
            train_arr, test_arr, _ = self.data_transformation.initiate_data_transformation(train_path, test_path)
            
            #step 3
            logger.info('STEP 3 - MODEL TRAINING')
            best_score, best_model_name, model_report = self.model_trainer.initiate_model_trainer(train_arr, test_arr)
            
            #Print FInal results
            logger.info('='*60)
            logger.info('TRAININF PIPELINE COMPLETED')
            logger.info(f'best model: {best_model_name}')
            logger.info(f'best score : {best_score:.4f}')
            logger.info('='*60)
            
            
            print('='*60)
            print('TRAININF PIPELINE COMPLETED')
            print(f'best model: {best_model_name}')
            print(f'best R2 : {best_score:.4f}')
            print('='*60)
            print('MODEL LEADERSHIP')
            
            for name, score in sorted(
                model_report.items(), key =lambda x: x[1], reverse = True):
                marker = '<- BEST 'if  name == best_model_name else "" 
                print(f'{name:<25} r2: {score:.4f}{marker}')
                
            return best_score, best_model_name,model_report    
        except Exception as e:
            raise CustomException(e,sys)