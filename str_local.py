import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from geopy.distance import geodesic
import time
import json
import os
import pickle
from IPython.display import display
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import xgboost as xgb
from typing import Dict, Any
import joblib
import random


# Load the trained model and encodings
model = joblib.load('best_trained_model.pkl')
building_encoding = joblib.load('State_of_building_encoding.pkl')
epc_encoding = joblib.load('epc_encoding.pkl')

# Define mappings for categorical features
category_map_building = {'To restore': 'To renovate', 'To be done up': 'To renovate', 'Just renovated': 'Good'}
category_map_epc = {'A+': 'A', 'A++': 'A', 'G': 'F'}

# Coordinates of the 1O major cities (example coordinates, use the correct ones for your case)
city_15_coords = pd.DataFrame({
    'City': ['Brussels', 'Antwerp', 'Ghent', 'Bruges', 'Liège', 'Namur', 'Leuven', 'Mons', 'Aalst', 'Sint-Niklaas'],
    'Latitude': [50.8503, 51.2194, 51.0543, 51.2093, 50.6293, 50.4811, 50.8794, 50.4542, 50.9402, 51.2170],
    'Longitude': [4.3517, 4.4025, 3.7174, 3.2247, 5.3345, 4.8708, 4.7004, 3.9460, 4.0710, 4.4155]
})

# Function to get coordinates from address
def get_coordinates(street, number, postal_code):
    geolocator = Nominatim(user_agent="house_price_app")
    try:
        location = geolocator.geocode(f"{number} {street}, {postal_code}, Belgium")
        if location:
            return location.latitude, location.longitude
        else:
            st.error("Could not find the location. Please check the address details.")
            return None, None
    except GeocoderTimedOut:
        st.error("Geocoding service timed out. Please try again.")
        return None, None

# Function to check if the house is within 15 km of any city
def is_within_15km_of_city(house_coords):
    for index, row in city_15_coords.iterrows():
        city_coords = (row['Latitude'], row['Longitude'])
        try:
            distance = geodesic(house_coords, city_coords).km
            if distance <= 15:
                return True, row['City']  # The house is within 15 km of this city
        except ValueError as e:
            st.error(f"Error calculating distance: {e}")
            return False, None
    return False, None  # Return False if no city is within 15 km

# Streamlit app definition
def main():
    st.title("House Price Prediction")

    # New address-based inputs
    street = st.text_input("Street")
    number = st.text_input("House Number")
    postal_code = st.text_input("Postal Code")

    # Input fields for other numerical features
    bedrooms = st.number_input("Number of Bedrooms", min_value=0)
    living_area = st.number_input("Living Area (sq meters)", min_value=0.0)
    swimming_pool = st.selectbox("Has Swimming Pool?", ["No", "Yes"])
    swimming_pool = 1 if swimming_pool == "Yes" else 0 
    facades = st.number_input("Number of Facades", min_value=1, max_value=9)
    land_surface = st.number_input("Land Surface (sq meters)", min_value=0.0)

    # Input fields for categorical features
    state_of_building = st.selectbox("State of Building", options=['Good', 'As new', 'To renovate', 'To be done up', 'Just renovated', 'To restore'])
    epc = st.selectbox("EPC Rating", options=['A++', 'A+', 'A', 'B', 'C', 'D', 'E', 'F', 'G'])

    # Get coordinates from address input if street, number, and postal code are provided
    if street and number and postal_code:
        latitude, longitude = get_coordinates(street, number, postal_code)
        if latitude and longitude:
            # Check if house is within 15 km of any city
            is_within_15, city_name = is_within_15km_of_city((latitude, longitude))
            if is_within_15:
                st.write(f"The house is within 15 km of {city_name}.")
                city_15 = 1  # Assigned to City 15
            else:
                st.write("The house is not within 15 km of any major city.")
                city_15 = 0  # Not Assigned to City 15
        else:
            city_15 = 0  # Default to 0 if geocoding failed
    else:
        city_15 = 0  # Default to 0 if address inputs are missing

    # Map user inputs for categorical fields to encoded values
    mapped_state = category_map_building.get(state_of_building, state_of_building)  # map using category map
    state_encoded = building_encoding.get(mapped_state, np.nan)  # encode using target encoding dictionary

    mapped_epc = category_map_epc.get(epc, epc)  # map using EPC category map
    epc_encoded = epc_encoding.get(mapped_epc, np.nan)  # encode using EPC target encoding dictionary

    # Check if encoding succeeded
    if np.isnan(state_encoded) or np.isnan(epc_encoded):
        st.error("Error: One of the encoded values for 'State of Building' or 'EPC' is missing in encoding dictionary.")
    else:
        # Collect inputs in a DataFrame
        input_data = pd.DataFrame({
            'Number_of_bedrooms': [bedrooms],
            'Living_area': [living_area],
            'Swimming_Pool': [swimming_pool],
            'Number_of_facades': [facades],
            'landSurface': [land_surface],
            'Has_Assigned_City_15': [city_15],  # Include the city_15 flag in the data
            'State_of_building_encoded': [state_encoded],
            'epc_encoded': [epc_encoded]
        })

        # Predict price
        if st.button("Predict Price"):
            predicted_price = model.predict(input_data)[0]
            st.write(f"### Predicted Price: €{predicted_price:,.2f}")

if __name__ == "__main__":
    main()