"""Use the saved preprocessor + model to predict quality for new wines."""
import pandas as pd

from src.config import FEATURES, MODEL_PATH, PREPROCESSOR_PATH
from src.data_transformation import add_features
from src.utils import load_object


class WineQualityPredictor:
    def __init__(self):
        self.preprocessor = load_object(PREPROCESSOR_PATH)
        self.model = load_object(MODEL_PATH)

    def predict(self, wines: pd.DataFrame):
        wines = add_features(wines)
        X = self.preprocessor.transform(wines[FEATURES])
        return self.model.predict(X)


if __name__ == "__main__":
    sample = pd.DataFrame([
        # High alcohol, low volatile acidity -> expected good wine
        {"fixed_acidity": 8.5, "volatile_acidity": 0.3, "citric_acid": 0.45, "residual_sugar": 2.2,
         "chlorides": 0.06, "free_sulfur_dioxide": 12, "total_sulfur_dioxide": 30, "density": 0.995,
         "ph": 3.25, "sulphates": 0.8, "alcohol": 12.5},
        # Low alcohol, high volatile acidity -> expected poor wine
        {"fixed_acidity": 7.4, "volatile_acidity": 0.9, "citric_acid": 0.0, "residual_sugar": 1.9,
         "chlorides": 0.09, "free_sulfur_dioxide": 11, "total_sulfur_dioxide": 50, "density": 0.998,
         "ph": 3.5, "sulphates": 0.5, "alcohol": 9.2},
    ])
    for wine, score in zip(sample.to_dict("records"), WineQualityPredictor().predict(sample)):
        print(f"alcohol {wine['alcohol']:>4}, volatile acidity {wine['volatile_acidity']} -> quality {score:.2f}")
