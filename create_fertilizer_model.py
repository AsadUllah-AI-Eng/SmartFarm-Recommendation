import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Read the dataset
df = pd.read_csv('Fertilizer Prediction.csv')

# Fix column names by stripping whitespace
df.columns = df.columns.str.strip()

# Create label encoders for categorical variables
soil_type_encoder = LabelEncoder()
crop_type_encoder = LabelEncoder()
fertilizer_encoder = LabelEncoder()

# Encode categorical variables
df['Soil Type'] = soil_type_encoder.fit_transform(df['Soil Type'])
df['Crop Type'] = crop_type_encoder.fit_transform(df['Crop Type'])
df['Fertilizer Name'] = fertilizer_encoder.fit_transform(df['Fertilizer Name'])

# Prepare features and target
X = df[['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous']]
y = df['Fertilizer Name']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
with open('fertilizer-recommendation-system.pkl', 'wb') as f:
    pickle.dump(model, f)

# Print model accuracy
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)
print(f"Training Accuracy: {train_accuracy:.2f}")
print(f"Testing Accuracy: {test_accuracy:.2f}")

# Save the encoders for later use
with open('soil_type_encoder.pkl', 'wb') as f:
    pickle.dump(soil_type_encoder, f)
with open('crop_type_encoder.pkl', 'wb') as f:
    pickle.dump(crop_type_encoder, f)

# Print the mapping for soil types and crop types
print("\nSoil Type Mapping:")
for i, label in enumerate(soil_type_encoder.classes_):
    print(f"{label}: {i}")

print("\nCrop Type Mapping:")
for i, label in enumerate(crop_type_encoder.classes_):
    print(f"{label}: {i}")

print("\nFertilizer Type Mapping:")
for i, label in enumerate(fertilizer_encoder.classes_):
    print(f"{label}: {i}") 