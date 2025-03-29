from flask import Flask, render_template
from model_utils import run_prediction_and_plot

app = Flask(__name__)

@app.route('/')
def index():
    run_prediction_and_plot()
    return render_template('index.html')

@app.route('/run_model')
def run_model():
    run_prediction_and_plot()
    return "Model run successfully!"
