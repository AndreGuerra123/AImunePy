from flask import Flask, request, jsonify
from flask_cors import CORS
from controllers.train import trainer
from controllers.predict import predictor
from controllers.evaluate import evaluator

app = Flask(__name__)
CORS(app)

# this will route the request to the an API wich performs modelling in tensorflow


@app.route("/train", methods=['POST'])
def train():
    return jsonify(trainer(request.json))

# this will route the request to the an API wich performs metrics evaluation in tensorflow


@app.route("/evaluate", methods=['POST'])
def evaluate():
    return jsonify(evaluator(request.json))

# this will route the request to the an API wich performs image prediciton in tensorflow


@app.route("/predict", methods=['POST'])
def predict():
    return jsonify(predictor(request.json))

# HTTP Errors handlers


@app.errorhandler(404)
def url_error(e):
    return jsonify(e)


@app.errorhandler(500)
def server_error(e):
    return jsonify(e)


# This will run this microserver in localhost port 5000

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
