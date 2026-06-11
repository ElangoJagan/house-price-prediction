from src.components.data_ingestion import dataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

# Step 1
ingestion = dataIngestion()
train_path, test_path = ingestion.initiate_data_ingestion()

# Step 2
transformation = DataTransformation()
train_arr, test_arr, _ = transformation.initiate_data_transformation(
    train_path, test_path
)

# Step 3
trainer = ModelTrainer()
best_score, best_model, report = trainer.initiate_model_trainer(
    train_arr, test_arr
)

print(f"\nBest Model : {best_model}")
print(f"Best Score : {best_score:.4f}")
print(f"\nFull Leaderboard:")
for name, score in sorted(report.items(), key=lambda x: x[1], reverse=True):
    print(f"  {name:<25} R2: {score:.4f}")