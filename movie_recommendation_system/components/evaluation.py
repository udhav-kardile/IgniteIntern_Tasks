from surprise import accuracy
from sklearn.metrics import mean_squared_error
import numpy as np

def evaluate(predictions):
    y_true = [pred.r_ui for pred in predictions]
    y_pred = [pred.est for pred in predictions]
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse
