import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
import joblib

def train(train_scaled):
    mlflow.set_tracking_uri("sqlite:///../mlflow.db")
    X_train = train_scaled[['HomePlanet', 'CryoSleep', 'Destination', 'VIP',
                            'Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
                            'Group_size', 'Family_size']]
    y_train = train_scaled["Transported"]

    with mlflow.start_run() as run:

        model = LogisticRegression(
            C=0.03,
            solver="liblinear",
            penalty="l2",
            max_iter=491
        )

        model.fit(X_train, y_train)

        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("C", 0.03)
        mlflow.log_param("solver", "liblinear")
        mlflow.log_param("penalty", "l2")
        mlflow.log_param("max_iter", 491)

        mlflow.sklearn.log_model(model, "model")

        joblib.dump(model, "artifacts/model.pkl")

        print("Model trained")
        return run.info.run_id

if __name__ == "__main__":
    train(None)