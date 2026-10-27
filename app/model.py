import os
import joblib
import pandas as pd

class ModelLoader:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        self.model = joblib.load(self.model_path)

    def predict(self, features_df: pd.DataFrame):
        prediction = self.model.predict(features_df)[0]
        if hasattr(self.model, "predict_proba"):
            probability = self.model.predict_proba(features_df)[0][1]
        else:
            probability = float(prediction)
        return int(prediction), float(probability)
