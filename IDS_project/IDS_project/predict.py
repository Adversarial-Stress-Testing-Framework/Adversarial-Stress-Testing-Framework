import joblib
import numpy as np

# ✅ load saved files
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

def predict(input_data):
    """
    input_data = list of 41 features (KDD dataset)
    """

    # convert to numpy
    data = np.array(input_data).reshape(1, -1)

    # apply SAME scaling
    data = scaler.transform(data)

    # predict
    prediction = model.predict(data)[0]

    return "Normal" if prediction == 0 else "Attack"