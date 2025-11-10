🌾 Integrated Crop and Fertilizer Recommendation System

💡 Overview

This project presents a comprehensive, integrated machine learning system designed to assist farmers and agronomists. It features two core functionalities:

Crop Recommendation: Suggests the optimal crop to cultivate based on seven key environmental and soil parameters (N, P, K, Temperature, etc.).

Fertilizer Recommendation: Predicts the most suitable fertilizer required for a specified crop, soil type, and nutrient levels (Temperature, Moisture, N, P, K, etc.).

The entire system is deployed as a single, centralized Flask web application for ease of use.

✨ Project Components

File Name

Role in Project

Description

app.py

Main Application

The Flask application entry point that handles routing, user input, model loading, and serving predictions for both systems.

Crop_recommendation.csv

Crop Dataset

Dataset used to train the Crop Recommendation Model.

Fertilizer Prediction.csv

Fertilizer Dataset

Dataset used to train the Fertilizer Recommendation Model.

Crop Recommendation Using Machine Learning.ipynb

Crop Notebook

Jupyter notebook for data cleaning, EDA, and training the Crop Recommendation model.

fertilizer-recommendation-system.ipynb

Fertilizer Notebook

Jupyter notebook for data preprocessing (encoding), training, and evaluation of the Fertilizer Recommendation model.

create_fertilizer_model.py

Model Creation Script

Python script used to train and save the fertilizer model and encoders.

model.pkl

Crop Model

Trained Random Forest Classifier for Crop Recommendation.

fertilizer-recommendation-system.pkl

Fertilizer Model

Trained Random Forest Classifier for Fertilizer Recommendation.

minmaxscaler.pkl

Scaler

MinMaxScaler object used for preprocessing input features for the Crop Model.

crop_type_encoder.pkl

Encoder

LabelEncoder used to encode/decode the categorical 'Crop Type' feature for the Fertilizer Model.

⚙️ Model Input Features

1. Crop Recommendation Inputs

The Crop Model predicts the best crop based on 7 numerical features:

N: Nitrogen ratio in soil

P: Phosphorus ratio in soil

K: Potassium ratio in soil

Temperature: Temperature in Celsius

Humidity: Relative humidity in %

pH: pH value of the soil

Rainfall: Rainfall in mm

2. Fertilizer Recommendation Inputs

The Fertilizer Model predicts the best fertilizer based on 8 features, including both numerical and categorical inputs:

Temparature

Humidity

Moisture

Soil Type (Categorical)

Crop Type (Categorical)

Nitrogen (N)

Potassium (K)

Phosphorous (P)

🚀 Installation & Setup

To run the Flask application locally, follow these steps:

1. Dependencies

The project requires the following Python libraries. Install them using pip:

pip install Flask numpy pandas scikit-learn



2. Run the Application

Execute the main application file:

python app.py



3. Access the App

Once the server is running, open your web browser and navigate to:

http://127.0.0.1:5000/