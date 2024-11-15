from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from geopy.geocoders import Nominatim
from geopy.distance import geodesic

app = FastAPI()

# Load the trained model and encodings
model = joblib.load('best_trained_model.pkl')
building_encoding = joblib.load('State_of_building_encoding.pkl')
epc_encoding = joblib.load('epc_encoding.pkl')

# Mapping dictionaries
category_map_building = {'To restore': 'To renovate', 'To be done up': 'To renovate', 'Just renovated': 'Good'}
category_map_epc = {'A+': 'A', 'A++': 'A', 'G': 'F'}

# Coordinates of major cities
city_15_coords = pd.DataFrame({
    'City': ['Brussels', 'Antwerp', 'Ghent', 'Bruges', 'Liège', 'Namur', 'Leuven', 'Mons', 'Aalst', 'Sint-Niklaas'],
    'Latitude': [50.8503, 51.2194, 51.0543, 51.2093, 50.6293, 50.4811, 50.8794, 50.4542, 50.9402, 51.2170],
    'Longitude': [4.3517, 4.4025, 3.7174, 3.2247, 5.3345, 4.8708, 4.7004, 3.9460, 4.0710, 4.4155]
})

# Request body model
class HouseData(BaseModel):
    street: str
    number: str
    postal_code: str
    bedrooms: int
    living_area: float
    swimming_pool: bool
    facades: int
    land_surface: float
    state_of_building: str
    epc: str

# Helper function to get coordinates from address
def get_coordinates(street, number, postal_code):
    geolocator = Nominatim(user_agent="house_price_app")
    location = geolocator.geocode(f"{number} {street}, {postal_code}, Belgium")
    if location:
        return location.latitude, location.longitude
    else:
        return None, None

# Helper function to check if house is within 15 km of a major city
def is_within_15km_of_city(house_coords):
    for _, row in city_15_coords.iterrows():
        city_coords = (row['Latitude'], row['Longitude'])
        distance = geodesic(house_coords, city_coords).km
        if distance <= 15:
            return True
    return False

@app.get("/")
def read_root():
    return {"message": "FastAPI is running!"}

@app.post("/predict")
def predict_price(data: HouseData):
    # Get coordinates
    latitude, longitude = get_coordinates(data.street, data.number, data.postal_code)
    if latitude is None or longitude is None:
        raise HTTPException(status_code=400, detail="Could not geocode the provided address.")

    # Determine if house is within 15km of a major city
    city_15 = 1 if is_within_15km_of_city((latitude, longitude)) else 0

    # Encode categorical features
    mapped_state = category_map_building.get(data.state_of_building, data.state_of_building)
    state_encoded = building_encoding.get(mapped_state, np.nan)
    mapped_epc = category_map_epc.get(data.epc, data.epc)
    epc_encoded = epc_encoding.get(mapped_epc, np.nan)

    # Validate encoding success
    if np.isnan(state_encoded) or np.isnan(epc_encoded):
        raise HTTPException(status_code=400, detail="Invalid state or EPC encoding.")

    # Prepare data for model input
    input_data = pd.DataFrame({
        'Number_of_bedrooms': [data.bedrooms],
        'Living_area': [data.living_area],
        'Swimming_Pool': [1 if data.swimming_pool else 0],
        'Number_of_facades': [data.facades],
        'landSurface': [data.land_surface],
        'Has_Assigned_City_15': [city_15],
        'State_of_building_encoded': [state_encoded],
        'epc_encoded': [epc_encoded]
    })

    # Predict price
    predicted_price = model.predict(input_data)[0]
    return {"predicted_price": predicted_price}
