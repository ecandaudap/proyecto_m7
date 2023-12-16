'''Modelo API'''
import joblib

def make_prediction(data):
    data = data["data"]
    input_data = data
    model = joblib.load("tpot_model.pkl")
    loaded_vectorizer = joblib.load("count_vectorizer.pkl")
    input_data = "".join(char for char in input_data if char not in ("?", ".", ";", ":", "!", '"', ","))
    new_data = loaded_vectorizer.transform([input_data])
    prediction = int(model.predict([new_data][0]))
   
    return {
        "value": [prediction],
    }
