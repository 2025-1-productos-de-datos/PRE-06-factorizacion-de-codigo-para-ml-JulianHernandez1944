def compare_models(current_model, best_model, x_test, y_test):
    """
    Compara el modelo actual con el mejor modelo guardado.
    """
    if best_model is None or current_model.score(x_test, y_test) > best_model.score(
        x_test, y_test
    ):
        return current_model
    return best_model
