import streamlit as st
import joblib
import numpy as np
import pandas as pd
import mlflow
from pathlib import Path

# Load preprocessor and model
encoders = joblib.load(Path(__file__).parent/"artifacts/preprocessor.pkl")
model  = joblib.load(Path(__file__).parent/"artifacts/model.pkl")

def main():
    st.title("ASG 04 MD - Leonardus Hasan - Spaceship Titanic Model Deployment")
    name            = st.text_input("Name", "Leonardus Hasan")
    home_planet     = st.selectbox("Home Planet", ("Earth", "Europa", "Mars"), index=1)
    cryo_sleep      = st.selectbox("Cryo Sleep", (True, False), index=0)
    destination     = st.selectbox("Destination", ("TRAPPIST-1e", "55 Cancri e", "PSO J318.5-22"), index = 2)
    vip             = st.selectbox("VIP", (True, False), index=0)
    age             = st.number_input("Age", min_value=0, value= 19)
    room_service    = st.slider("Room Service", min_value=0.0, value= 7000.0)
    food_court      = st.slider("Food Court", min_value=0.0, value= 15000.0)
    shopping_mall   = st.slider("Shopping Mall", min_value=0.0, value= 12000.0)
    spa             = st.slider("Spa", min_value=0.0, value= 10000.0)
    vr_deck         = st.slider("VR Deck", min_value=0.0, value= 12000.0)
    group_size      = st.number_input("Group Size", min_value=0, max_value=10, value= 2)
    family_size     = st.number_input("Family Size", min_value=0, max_value=10, value= 2)


    if st.button("Make Prediction"):
        features = [[home_planet, cryo_sleep, destination, vip, age, room_service, food_court,
                     shopping_mall, spa, vr_deck, group_size, family_size]]
        features = pd.DataFrame(features, columns=['HomePlanet', 'CryoSleep', 'Destination', 'VIP',
                            'Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
                            'Group_size', 'Family_size'])
        result = make_prediction(features)
        if result == 1:
            st.success("✅️ Passenger will be transported")
        else:
            st.error("❌ Passenger will NOT be transpoted")


def make_prediction(features):
    for col, encoder in encoders.items():
        if col in features.columns:
            features[col] = encoder.transform(features[col].astype(str))

    prediction = model.predict(features)
    return prediction[0]


if __name__ == "__main__":
    main()