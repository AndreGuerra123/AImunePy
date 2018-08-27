from flask import Flask, request, jsonify
from flask_cors import CORS

from controllers.train import Trainer
from controllers.predict import Predictor
from controllers.results import Resulter
from bokeh.embed import components

app = Flask(__name__)
CORS(app)

# this will route the request to the an API wich performs model training, saves weights and resutls

@app.route("/train", methods=['POST'])
def train():
    Trainer(request.json)
    return jsonify("Train finished.")
    # this will route the request to the an API wich performs image prediciton in tensorflow

@app.route("/results",methods=['POST'])
def result():
    plots = Resulter(request.json).getHtml()
    script, div = components(plots, wrap_script=False)
    return jsonify({'div': div, 'script': script})

@app.route("/predict", methods=['POST'])
def predict():
    return jsonify(Predictor(request.json))


# HTTP Errors handlers

@app.errorhandler(404)
def url_error(e):
    return jsonify(e)


@app.errorhandler(500)
def server_error(e):
    return jsonify(e)


# This will run this microserver in localhost port 5000

if __name__ == '__main__':
    app.run(host='127.0.0.1',port=5000,debug=True)