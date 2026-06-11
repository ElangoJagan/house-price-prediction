import os
from symtable import Class
import sys
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor)
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score


from src.logger import logger
from src.exception import CustomException
from src.utils import Utils

@dataclass
class ModelTrainerConfig:
    """Stores model save path loaded from config.yaml."""
    model_path:str = ""
    r2_score_threshold: float = 0.6

class ModelTrainer:
    """
    Handles model training and selection.
    - Trains 9 models with GridSearchCV tuning
    - Compares all models by R2 score
    - Saves best model to disk
    """
    def __init__(self):
        config= Utils.read_yaml('config/config.yaml')
        mt_config = config['model_trainer']
        
        self.model_trainer_config = ModelTrainerConfig(model_path = mt_config['model_path'], r2_score_threshold= mt_config['r2_threshold'])
        
    
    def _get_models(self):
        """Returns dictionary of all models to train."""
        return{
            'Linear Regression':LinearRegression(),
            'Ridge regression': Ridge(),
            'Lasso Regression': Lasso(),
            'Decision Tree':DecisionTreeRegressor(),
            'Random Forest ': RandomForestRegressor(),
            'GradientBoosting ': GradientBoostingRegressor(),
            'AdaBoost': AdaBoostRegressor(),
            'XGBoost':XGBRegressor(verbose = 0),
            'catboost':CatBoostRegressor(verbose = False)
        }
    
    def _get_params(self) -> dict:
        return Utils.read_yaml('config/params.yaml')
    
    def initiate_model_trainer(self, train_array, test_array):
        """
        Trains all models, selects best by R2 score,
        saves it and returns final score.

        Returns:
            tuple: (best_r2_score, best_model_name, full_report)
            
        """
        logger.info('>>> Model Training Started' )
        try:
            # Split arrays into x and y 
            x_train, y_train = train_array[:, :-1], train_array[:,-1]
            x_test, y_test= test_array[:,:-1],test_array[:,-1]
            
            logger.info(f'x_train: {x_train.shape}| y_train:{y_train.shape}')
            logger.info(f'x_test:{x_test.shape} | y_test:{y_test.shape}')
            
            #step 2 - get models and params
            models = self._get_models()
            params = self._get_params()
            
            # step 3 - train and evaluate all models
            model_report = Utils.evaluate_models(
                x_train = x_train, y_train= y_train, x_test=x_test, y_test= y_test,models = models,params = params
            )
            
            #Step 4 Print leaderbaord 
            
            logger.info('='*50)
            logger.info('Model LeaderBoard')
            
            for name, score in sorted(
                model_report.items(), key=lambda x: x[1], reverse = True
            ):
                logger.info(f'{name:<25} R2:{score:.4f}')
            logger.info('='*50)
        
            #Step 5 - pick best model
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]
            
            logger.info(f'best model :{best_model_name}')
            logger.info(f'best_score: {best_model_score:.4f}')
            
            #step 6 - check quality threshold
            
            if best_model_score< self.model_trainer_config.r2_score_threshold:
                raise CustomException(
                    f'No Good model found. Best  R2: {best_model_score:.4f}'
                    f'is below threshold {self.model_trainer_config.r2_score_threshold}', sys
                )
            # Step 7 save obj
            
            Utils.save_object(
                file_path = self.model_trainer_config.model_path,
                obj = best_model
            )
            
            logger.info(f'best  model is saved at {self.model_trainer_config.model_path}')
            logger.info(">>> Model Training completed")
            
            return (
                best_model_score, best_model_name, model_report
            )
        except Exception as e:
            raise CustomException(e,sys)