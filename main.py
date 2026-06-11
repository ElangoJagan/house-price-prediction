import sys 
import argparse
from src.logger import logger
from src.exception import CustomException
from src.pipeline.train_pipeline import TrainPipeline
from src.pipeline.predict_pipeline import  PredictPipeline, CustomData

def train():
    '''Runs the full training pipeline'''
    try:
        logger.info('Training Mode selected ')
        pipeline= TrainPipeline()
        pipeline.run()
    
    except Exception as e:
        raise CustomException(e,sys)

def predict():
    """Runs prediction on a sample input."""
    try:
        logger.info('Prediction mode selected')
        
        #sample house input
        data = CustomData( 
                MedInc=3.5,
                HouseAge=20.0,
                AveRooms=5.2,
                AveBedrms=1.1,
                Population=800.0,
                AveOccup=2.8,
                Latitude=34.05,
                Longitude=-118.25
        )
        
        #convert to dataframe
        df = data.get_data_as_dataframe()
        print('/nInput Features')
        print(df.to_string(index= False))
        
        #Predict
        pipeline = PredictPipeline()
        prediction = pipeline.predict(df)
        
        print(f"\nPredicted House Price: ${prediction[0] * 100_000:,.0f}")
        logger.info(f"Prediction: ${prediction[0] * 100_000:,.0f}")

    except Exception as e:
        raise CustomException(e, sys)

if __name__== "__main__":
    #Argument Parser - decides  train or predict mode
    parser = argparse.ArgumentParser(description = 'House Price Prediction')
    parser.add_argument("--predict",action= 'store_true', help= 'Run Prediction pipeline instead of training ')
    args = parser.parse_args()
    
    if args.predict:
        predict()
    else:
        train()