import os 
import sys
import dill 
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.logger import logger
from src.exception import CustomException

class utils:
    
     """
    Reusable utility functions used across all components.
    - save_object   : serialize and save any Python object
    - load_object   : load a saved Python object
    - evaluate_models : train, tune and evaluate multiple models
    - read_yaml     : read yaml config files
    """
    
    
    @staticmethod
    def save_object(file_path: str, obj)-> None :
        """Saves any Python object to disk using dill."""
        try:
            logger.info(f'saving object to {file_path}')
            dir_path = os.path.dirname(file_path)
            os.makedirs(dir_path, exist_ok= True)
            
            with open(file_path, 'wb') as file_obj:
                dill.dump(obj, file_obj)
            
            logger.info('Object Saved Successfully')
        
        except Exception as e:
            raise CustomException(e, sys) 
        
    @staticmethod
    def load_object(file_path:str):
        'Loads a stored python obj from the folder '
        try:
            logger.info('Load python object from {file_path}')
            
            with open(file_path, 'rb') as file_obj:
                obj = dill.load(file_obj)   
            
            logger.info('Obj successsfully loaded ')
            return obj
        except Exception as e:
            raise CustomException(e,sys)
        
    @staticmethod
    def evaluate_models(x_train, y_train, x_test, y_test, models:dict, params:dict) :
        """
        Trains each model with GridSearchCV tuning.
        Returns dict of model_name -> test R2 score.
        """
        try:
            report = {}
            
            for model, model_name in models.items():
                logger.info('Training model is {model_name}')
                param_grid = params.get(model_name, {})
                
                if param_grid:
                    gs = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
                    gs.fit(x_train, y_train)
                    model.set_params(**gs.best_params_)
                    logger.info(f'best params for {model_name}: {gs.best_params_}')
                    
                model.fit(x_train, y_train)
                
                y_train_pred = model.predict(x_train)
                y_test_pred = model.predict(x_test)
                
                train_r2 = r2_score(y_train, y_train_pred)
                test_r2 = r2_score(y_test, y_test_pred)
                
                logger.info(f'{model_name} Train_r2:{train_r2:.4f} | test_r2:.{test_r2:.4f}')
                report[model_name] = test_r2
                
            return report 
        except Exception as e:
            raise CustomException(e,sys)
        
    @staticmethod
    def read_yaml(file_path:str):
            """Reads a yaml file and returns contents as dictionary."""
        try:
            logger.info('reading the file ')
            with open(file_path, 'r') as f:
                content = yaml.safe_load(f)
                
            logger.info('YAML file read successfully')
            return content 
        
        except Exception as e:
            raise CustomException(e,sys)
        