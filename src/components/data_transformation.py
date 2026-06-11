import os 
import pandas as pd
import sys
import numpy as np
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from src.logger import logger
from src.exception import CustomException
from src.utils import Utils


@dataclass
class DataTransformationConfig:
    """Stores preprocessor path loaded from config.yaml."""
    preprocessor_path: str = ""


class DataTransformation:
    """
    Handles data transformation.
    - Builds preprocessing pipeline
    - Fits on train, transforms both train & test
    - Saves preprocessor object to disk
    """

    def __init__(self):
        config = Utils.read_yaml("config/config.yaml")
        dt_config = config["data_transformation"]

        self.transformation_config = DataTransformationConfig(
            preprocessor_path=dt_config["preprocessor_path"]
        )

    def get_data_transformer_object(self, numerical_columns: list):
        """
        Builds and returns ColumnTransformer pipeline.
        Add categorical pipeline here for future projects.
        """
        try:
            logger.info(f"Numerical columns: {numerical_columns}")

            # Pipeline for numerical features
            numerical_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            # ColumnTransformer — apply pipeline to correct columns
            preprocessor = ColumnTransformer(transformers=[
                ("numerical_pipeline", numerical_pipeline, numerical_columns)
            ])

            logger.info("Preprocessing pipeline created successfully")
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        """
        Applies preprocessing to train and test data.

        Returns:
            tuple: (train_array, test_array, preprocessor_path)
        """
        logger.info(">>>>>> Data Transformation Started")
        try:
            # Step 1 — Load train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logger.info(f"Train shape: {train_df.shape} | Test shape: {test_df.shape}")

            # Step 2 — Separate features and target
            target_column = "MedHouseVal"

            x_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]

            x_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            logger.info(f"Target column: {target_column}")
            logger.info(f"Feature columns: {list(x_train.columns)}")

            # Step 3 — Get numerical columns
            numerical_columns = x_train.select_dtypes(exclude="object").columns.tolist()

            # Step 4 — Build preprocessor
            preprocessor = self.get_data_transformer_object(numerical_columns)

            # Step 5 — Fit on train, transform both
            x_train_scaled = preprocessor.fit_transform(x_train)
            x_test_scaled = preprocessor.transform(x_test)

            logger.info(f"x_train_scaled shape: {x_train_scaled.shape}")
            logger.info(f"x_test_scaled shape: {x_test_scaled.shape}")

            # Step 6 — Combine features + target back together
            train_array = np.c_[x_train_scaled, np.array(y_train)]
            test_array = np.c_[x_test_scaled, np.array(y_test)]

            # Step 7 — Save preprocessor
            Utils.save_object(
                file_path=self.transformation_config.preprocessor_path,
                obj=preprocessor
            )

            logger.info(">>>>>> Data Transformation completed")

            return (
                train_array,
                test_array,
                self.transformation_config.preprocessor_path
            )

        except Exception as e:
            raise CustomException(e, sys)