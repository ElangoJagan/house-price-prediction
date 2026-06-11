import os
from symtable import Class
import sys
from dataclasses import dataclass

from sklearn.linear_model import linearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor)
from xgboost import XGBRegressor, XGBoostRegressor
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
            
