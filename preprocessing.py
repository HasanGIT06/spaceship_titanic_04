import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os
import joblib

def preprocess_data(df):
    """Preprocess the data for modeling"""
    df = df.copy()
    
    # Define features to use
    categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'Age_group']
    numerical_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
                         'Cabin_num', 'Group_size', 'Solo', 'Family_size', 'TotalSpending',
                         'HasSpending', 'NoSpending', 'Age_missing', 'CryoSleep_missing'] + \
                        [col for col in df.columns if '_ratio' in col]
    
    # Fill missing values
    for col in categorical_features:
        df[col] = df[col].fillna('Unknown')
    
    for col in numerical_features:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    # Encode categorical features
    label_encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    # Select features
    feature_columns = categorical_features + numerical_features
    X = df[feature_columns]
    y = df['Transported'].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)
    train_scaled = pd.concat([X_train, pd.Series(y_train, name="Transported")], axis = 1)
    test_scaled = pd.concat([X_test, pd.Series(y_test, name="Transported")], axis = 1)
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(label_encoders, "artifacts/preprocessor.pkl")
    return train_scaled, test_scaled

if __name__ == "__main__":
    preprocess_data(None)