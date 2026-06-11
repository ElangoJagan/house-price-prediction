import sys
import pandas as pd
from dataclasses import dataclass 

from src.logger import logger
from src.exception import CustomException
from src.utils import Utils


class PredictPipeline:
    """
    Loads saved preprocessor and model.
    Preprocesses new input and returns prediction.
    """
    
    def __init__(self):
        config = Utils.read_yaml('config/config.yaml')
        self.model_path = config['model_trainer']['model_path']
        self.preprocessor_path = config['data_transformation']['preprocessor_path']
    
    def predict(self, features: pd.DataFrame):
        """
        Takes raw input features, preprocesses and predicts.

        Args:
            features: pd.DataFrame with raw input values

        Returns:
            np.ndarray: predicted house prices
        """
        try:
            logger.info("Loading model and preprocessor")
            
            
            #Load save Objects
            model = Utils.load_object(self.model_path)
            preprocessor = Utils.load_object(self.preprocessor_path)
            
            #scale  input  using  saved preprocessor
            data_scaled = preprocessor.transform(features)
            
            #predict
            prediction = model.predict(data_scaled)
            
            logger.info(f"'prediction done:'{prediction}")
            return prediction
        except Exception as e:
            raise CustomException(e,sys)
@dataclass
class CustomData:
    
    """
    Maps raw user inputs to a DataFrame.
    Each field matches a feature column from training.
    California Housing features.
    
    """
    MedInc:float
    HouseAge:float
    AveRooms: float     # Average rooms
    AveBedrms: float    # Average bedrooms
    Population: float   # Population
    AveOccup: float     # Average occupancy
    Latitude: float     # Latitude
    Longitude: float
    
    def get_data_as_dataframe(self) ->pd.DataFrame:
        """Converts user input into DataFrame for prediction."""
        try:
            data ={
                "MedInc": [self.MedInc],
                "HouseAge": [self.HouseAge],
                "AveRooms": [self.AveRooms],
                "AveBedrms": [self.AveBedrms],
                "Population": [self.Population],
                "AveOccup": [self.AveOccup],
                "Latitude": [self.Latitude],
                "Longitude": [self.Longitude]
            }
            df = pd.DataFrame(data)
            logger.info(f'input dataframe ceated : {df}')
            return df
        except Exception as e:
            raise CustomException(e,sys)
        