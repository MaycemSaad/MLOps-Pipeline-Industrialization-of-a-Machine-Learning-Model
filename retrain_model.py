import time
from model_pipeline import prepare_data, train_model, save_model


def retrain():
    """
    Function to retrain the model with the latest data.
    This will load the data, train the model, and save the retrained model.
    """
    print("Starting model retraining...")

    # Load the data and split into training and testing sets
    X_train, X_test, y_train, y_test = prepare_data("churn-bigml-80.csv")
    print("Data loaded successfully and split into train/test.")

    # Train the model using the training data
    model = train_model(X_train, y_train)
    print("Model trained successfully.")

    # Save the retrained model
    save_model(model, "retrained_model.joblib")
    print("Model retrained and saved successfully.")


if __name__ == "__main__":
    retrain()
