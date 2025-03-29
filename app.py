from flask import Flask, render_template
from model_utils import run_prediction_and_plot

app = Flask(__name__)

@app.route('/')
def index():
    image_path = "static/plot.png"
    run_prediction_and_plot()
    return render_template("index.html", image_path=image_path)

@app.route('/run_model')
def run_model():
    run_prediction_and_plot()
    return "Model run successfully and plot updated!"
