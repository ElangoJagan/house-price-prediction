from src.components.data_ingestion import dataIngestion
from src.components.data_transformation import DataTransformation

# Step 1
ingestion = dataIngestion()
train_path, test_path = ingestion.initiate_data_ingestion()

# Step 2
transformation = DataTransformation()
train_arr, test_arr, preprocessor_path = transformation.initiate_data_transformation(
    train_path, test_path
)

print(f"Train array shape : {train_arr.shape}")
print(f"Test array shape  : {test_arr.shape}")
print(f"Preprocessor saved: {preprocessor_path}")