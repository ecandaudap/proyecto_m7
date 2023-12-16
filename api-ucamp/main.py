from flask import (
    Flask,
    request,
    jsonify
)

from model import make_prediction

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    results = make_prediction(data)

    return jsonify({
        "results": results,
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )

