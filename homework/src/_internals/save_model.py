import os
import pickle


def save_model(model, save_path="models/estimator.pkl"):
    """
    Guarda el modelo en un archivo.
    """
    if not os.path.exists("models"):
        os.makedirs("models")

    with open(save_path, "wb") as file:
        pickle.dump(model, file)
