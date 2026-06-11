from src.pipeline.predict_pipeline import PredictPipeline, CustomData

# Sample house details
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

# Convert to DataFrame
df = data.get_data_as_dataframe()
print("Input:")
print(df)

# Predict
pipeline = PredictPipeline()
prediction = pipeline.predict(df)

# California Housing target is in $100,000 units
print(f"\nPredicted House Price: ${prediction[0] * 100_000:,.0f}")