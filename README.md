# immo-eliza-deployment

API to access the predictions of the immo-eliza-ml project

This project consists of two components:

FastAPI backend for processing house data and making predictions.

Streamlit frontend for user-friendly interaction with the application.
The application uses a trained machine learning model to predict house prices based on user inputs, leveraging geolocation data and other house features.

# Project Structure

fast.py: FastAPI backend to handle prediction requests.
str.py: Streamlit frontend for user interaction.
Dockerfile: Docker configuration to build and run the application.
requirements.txt: List of dependencies for the project.
best_trained_model.pkl: Trained model file for house price prediction.
State_of_building_encoding.pkl: Encoding file for building state.
epc_encoding.pkl: Encoding file for EPC (Energy Performance Certificate).

# Features

Predict house prices based on features such as location, size, number of facades, EPC rating, etc.

Uses geolocation to determine proximity to major cities.

Interactive Streamlit interface for user inputs.

REST API powered by FastAPI for backend processing.

# Requirements

Software
Docker
Render account (for deployment)

# Libraries

All required libraries are listed in requirements.txt. They include:

fastapi
uvicorn
streamlit
pandas
numpy
joblib
geopy
scikit-learn
Local Setup and Testing

1. Clone the Repository

bash

git clone <repository_url>
cd <repository_folder>

2. Install Dependencies

Use pip to install the required packages:

bash
pip install -r requirements.txt

3. Run Locally

Run FastAPI:

bash
uvicorn fast:app --host 0.0.0.0 --port 8000

Run Streamlit:

bash
streamlit run str.py --server.port=10000 --server.address=0.0.0.0

Access:

FastAPI: http://127.0.0.1:8000/docs
Streamlit: http://127.0.0.1:10000

# Deploying with Render

1. Create a New Web Service

Go to the Render dashboard.

Click New + > Web Service.

Connect your repository containing the project.

2. Set Deployment Options

Environment: Docker

Start Command:

bash

sh -c "uvicorn fast:app --host 0.0.0.0 --port 8000 & streamlit run str.py --server.port=10000 --server.address=0.0.0.0"

3. Deployment

Render will:

Build your Docker container using the Dockerfile.
Deploy both FastAPI and Streamlit apps.

4. Access the Application

Once deployed, Render provides a public URL. The Streamlit app will be accessible at:

https://<your-app-name>.onrender.com

# Files Explained

fast.py: FastAPI app providing an endpoint /predict to handle house price predictions.

str.py: Streamlit app for collecting user inputs and displaying predictions.

Dockerfile: Used to containerize the application for deployment.

requirements.txt: Lists all dependencies required to run the project.

.pkl Files: Model and encoding files used for making predictions.

# Example Usage

# Using Streamlit Interface

Open the Streamlit URL.

Input the house details, including address, number of bedrooms, living area, etc.

Click the Predict button.

View the predicted price.

# Using FastAPI Directly

Open FastAPI docs at /docs.

Use the /predict endpoint to make a prediction by providing JSON input:

{
"street": "Some Street",
"number": "123",
"postal_code": "1000",
"bedrooms": 3,
"living_area": 150,
"swimming_pool": true,
"facades": 4,
"land_surface": 200,
"state_of_building": "Good",
"epc": "A+"
}

Notes

Geolocation Limitations: The app relies on the Geopy library for geocoding. Ensure internet access for geocoding to work.

Major Cities: The app calculates the proximity of the house to a predefined list of major cities. You can update the city_15_coords DataFrame in both fast.py and str.py to customize this list.
