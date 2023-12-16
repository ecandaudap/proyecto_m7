'''Program used to detect sentiments in comments about a Game in Google Play Store'''

import joblib

model = joblib.load("tpot_model.pkl")

loaded_vectorizer = joblib.load("count_vectorizer.pkl")

input_data = ["AB for me has been really good in terms of learning, in all aspects, but there are some parts that I wish that could be changed. \
In the newest update, they removed the bar telling you how long of that specific achievement you need to advance to the next level \
(i.e. early riser achievement) so I wish that could be added back so I can keep track of my progress. \
But overall, 4 stars."]

input_data = "".join(char for char in input_data if char not in ("?", ".", ";", ":", "!", '"', ","))


new_data = loaded_vectorizer.transform([input_data])


prediction = model.predict(new_data)

SENTIMENT = ""

if prediction == 0:
    SENTIMENT = "negativo"
elif prediction == 1:
    SENTIMENT = "neutral"
elif prediction == 2:
    SENTIMENT = "positivo"

print(f'El sentimiento del comentario es {SENTIMENT}')
