from src.components.data_ingestion import dataIngestion

obj = dataIngestion()
train_path, test_path = obj.initiate_data_ingestion()

print(f"Train: {train_path}")
print(f"Test : {test_path}")