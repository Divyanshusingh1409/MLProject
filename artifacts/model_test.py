import pickle
import numpy as np
import pandas as pd

# Define file paths
MODEL_PATH = "artifacts/model.pkl"
PREPROCESSOR_PATH = "artifacts/preprocessor.pkl"

# Sample input data (Modify this based on your dataset)
sample_data = {
    "gender": ["male"],
    "race_ethnicity": ["group A"],
    "parental_level_of_education": ["bachelor's degree"],
    "lunch": ["free/reduced"],
    "test_preparation_course": ["completed"],
    "reading_score": [67],
    "writing_score": [68]
}

# Convert sample data to DataFrame
input_df = pd.DataFrame(sample_data)

# Load preprocessor
try:
    with open(PREPROCESSOR_PATH, "rb") as f:
        preprocessor = pickle.load(f)
    print("✅ Preprocessor loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load preprocessor: {e}")
    exit()

# Transform the input data
try:
    transformed_data = preprocessor.transform(input_df)
    print("✅ Data transformed successfully!")
except Exception as e:
    print(f"❌ Data transformation failed: {e}")
    exit()

# Load model
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    exit()

# Make a prediction
try:
    prediction = model.predict(transformed_data)
    print(f"✅ Model Prediction: {prediction}")
except Exception as e:
    print(f"❌ Model prediction failed: {e}")
