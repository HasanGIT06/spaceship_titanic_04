import pandas as pd
from feature_engineering import feature_engineering
from data_ingestion import ingest_data
from preprocessing import preprocess_data
from train import train
from evaluation import evaluate

def run_pipeline():
    print("=" * 50)
    print("Step 1: Data Ingestion")
    ingest_data()

    df = pd.read_csv("ingested/train.csv")
    print("Step 2: Feature Engineering")
    df = feature_engineering(df)

    print("\nStep 3: Preprocessing")
    train_scaled, test_scaled = preprocess_data(df)

    print("\nStep 4: Training")
    run_id = train(train_scaled)

    print("\nStep 5: Evaluation")
    accuracy, precision, recall = evaluate(test_scaled, run_id)

if __name__ == "__main__":
    run_pipeline()