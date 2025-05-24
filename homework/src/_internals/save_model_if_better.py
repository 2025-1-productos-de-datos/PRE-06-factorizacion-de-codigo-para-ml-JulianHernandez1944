import os
import pickle

from homework.src._internals.compare_models import compare_models
from homework.src._internals.save_model import save_model


def save_model_if_better(model, x_test, y_test, save_path="models/estimator.pkl"):
    """
    Guarda el modelo si es mejor que el modelo guardado anteriormente.
    """
    # Cargar el modelo guardado anteriormente
    best_model = None
    if os.path.exists(save_path):
        with open(save_path, "rb") as file:
            best_model = pickle.load(file)
    best_model = compare_models(model, best_model, x_test, y_test)
    save_model(best_model, save_path)
